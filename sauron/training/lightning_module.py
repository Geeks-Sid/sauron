# lightning_module.py

# sauron/training/lightning_module.py (New File)

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sauron.losses.surv_loss import (  # Assuming these are accessible
    CoxSurvLoss,
    CrossEntropySurvLoss,
    NLLSurvLoss,
)
from sauron.utils.generic_utils import calculate_error  # Assuming accessible
from sauron.utils.metrics import (  # Assuming accessible
    _calculate_classification_auc,
    _calculate_survival_c_index,
)
from sauron.utils.optimizers import get_optim  # Assuming accessible


class SauronLightningModule(pl.LightningModule):
    def __init__(self, model: nn.Module, args: Any):
        super().__init__()
        self.model = model
        # Use save_hyperparameters to make args accessible via self.hparams
        # and log them. args should be a Namespace or a dict.
        self.save_hyperparameters(
            args, logger=True
        )  # logger=True to log them with PL loggers

        self.loss_fn = self._get_loss_fn()

        self.train_survival_loss_alpha = (
            self.hparams.alpha_surv
            if self.hparams.task_type == "survival"
            and hasattr(self.hparams, "alpha_surv")
            else 0.0
        )
        # For evaluation, Cox and NLL/CE typically use alpha=0 for loss calculation
        # unless specifically designed otherwise. The original code used 0.0.
        self.eval_survival_loss_alpha = 0.0

        self.lambda_reg = getattr(self.hparams, "lambda_reg", 0.0)
        # L1 regularization example (can be adapted for L2)
        self.reg_type = getattr(self.hparams, "reg_type", None)

        # To store patient-level results from the test set
        self.test_patient_results = []  # List to store per-batch results
        self.test_patient_results_aggregated: Optional[Dict[str, Any]] = None

    def _get_loss_fn(self) -> nn.Module:
        """Initializes the loss function based on hparams."""
        task_type = self.hparams.task_type
        if task_type == "classification":
            return nn.CrossEntropyLoss()
        elif task_type == "survival":
            alpha_surv = getattr(self.hparams, "alpha_surv", 0.0)  # Used for training
            bag_loss = self.hparams.bag_loss
            if bag_loss == "ce_surv":
                return CrossEntropySurvLoss(
                    alpha=alpha_surv
                )  # alpha here is for training
            elif bag_loss == "nll_surv":
                return NLLSurvLoss(alpha=alpha_surv)  # alpha here is for training
            elif bag_loss == "cox_surv":
                return (
                    CoxSurvLoss()
                )  # Cox usually doesn't use alpha in its primary formulation
            else:
                raise NotImplementedError(f"Survival loss {bag_loss} not implemented.")
        else:
            raise ValueError(f"Invalid task_type: {task_type}")

    def forward(self, x: torch.Tensor) -> Tuple:
        """Forward pass through the model."""
        return self.model(x)

    def _unpack_batch_data(
        self, batch_data: Tuple
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[Any], Optional[torch.Tensor], Optional[Any]
    ]:
        """Unpacks batch data. Returns (model_input, label, event_time_cpu, censorship_gpu, slide_id (if available))."""
        slide_id = None
        if self.hparams.task_type == "classification":
            if len(batch_data) == 2:
                data, label = batch_data
            elif len(batch_data) == 3:  # Assuming (data, label, slide_id)
                data, label, slide_id = batch_data
            else:
                raise ValueError(
                    f"Unexpected batch data length for classification: {len(batch_data)}"
                )
            # Device placement is handled by Lightning for batch tensors
            return data, label, None, None, slide_id

        elif self.hparams.task_type == "survival":
            if len(batch_data) == 5:  # data_WSI, _omic, label, event_time, c
                data_WSI, _omic, label, event_time, c = batch_data
            elif (
                len(batch_data) == 6
            ):  # data_WSI, _omic, label, event_time, c, slide_id
                data_WSI, _omic, label, event_time, c, slide_id = batch_data
            else:
                raise ValueError(
                    f"Unexpected batch data length for survival: {len(batch_data)}"
                )

            event_time_cpu = (
                event_time.cpu().numpy()
                if isinstance(event_time, torch.Tensor)
                else event_time
            )
            return data_WSI, label, event_time_cpu, c, slide_id
        else:
            raise ValueError(f"Unknown task_type: {self.hparams.task_type}")

    def _calculate_main_loss(
        self,
        model_outputs: Tuple,
        label: torch.Tensor,
        c_tensor: Optional[torch.Tensor],
        current_survival_alpha: float,
    ) -> torch.Tensor:
        """Calculates the main loss component."""
        if self.hparams.task_type == "classification":
            logits, _, _, _, _ = model_outputs  # Assuming model output structure
            return self.loss_fn(logits, label)
        elif self.hparams.task_type == "survival":
            hazards, S, _, _, _ = model_outputs  # Assuming model output structure
            # Pass alpha to survival loss function. The loss function itself
            # should handle how alpha is used (e.g., NLLSurvLoss).
            # CoxSurvLoss might ignore alpha.
            if isinstance(self.loss_fn, (NLLSurvLoss, CrossEntropySurvLoss)):
                return self.loss_fn(
                    hazards=hazards,
                    S=S,
                    Y=label,
                    c=c_tensor,
                    alpha=current_survival_alpha,
                )
            elif isinstance(self.loss_fn, CoxSurvLoss):
                return self.loss_fn(
                    hazards=hazards, S=S, Y=label, c=c_tensor
                )  # Cox might not take alpha
            else:  # Fallback for other survival losses that might not have alpha explicitly
                return self.loss_fn(hazards=hazards, S=S, Y=label, c=c_tensor)

        raise ValueError(
            f"Unknown task_type for loss calculation: {self.hparams.task_type}"
        )

    def _get_regularization_loss(self) -> torch.Tensor:
        """Calculates L1 or L2 regularization loss if enabled."""
        reg_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_reg > 0 and self.reg_type:
            if self.reg_type.lower() == "l1":
                for param in self.model.parameters():
                    if param.requires_grad:
                        reg_loss += torch.sum(torch.abs(param))
            elif self.reg_type.lower() == "l2":
                for param in self.model.parameters():
                    if param.requires_grad:
                        reg_loss += torch.sum(param.pow(2))
            else:
                # warnings.warn(f"Unsupported regularization type: {self.reg_type}")
                pass  # Or raise error
        return reg_loss * self.lambda_reg

    def _common_step(self, batch: Tuple, batch_idx: int, stage: str) -> Dict[str, Any]:
        """Common logic for training, validation, and test steps."""
        model_input, label_gpu, event_time_cpu, c_gpu, slide_id_b = (
            self._unpack_batch_data(batch)
        )

        model_outputs = self(model_input)  # Calls forward

        current_alpha = (
            self.train_survival_loss_alpha
            if stage == "train"
            else self.eval_survival_loss_alpha
        )
        main_loss = self._calculate_main_loss(
            model_outputs, label_gpu, c_gpu, current_alpha
        )

        output_dict = {"main_loss": main_loss, "slide_ids": slide_id_b}

        if self.hparams.task_type == "classification":
            logits, y_prob, y_hat, _, _ = model_outputs
            output_dict.update(
                {
                    "preds": y_hat.detach(),  # For error calculation
                    "probs": y_prob.detach(),  # For AUC
                    "labels": label_gpu.detach(),
                    "logits_batch": logits.detach(),  # For patient results
                }
            )
        elif self.hparams.task_type == "survival":
            hazards, S, _, _, _ = model_outputs
            risk = -torch.sum(
                S, dim=1
            ).detach()  # Higher S sum means lower risk (longer survival)
            output_dict.update(
                {
                    "risks": risk,
                    "labels": label_gpu.detach(),  # Discrete time bin
                    "censorships": c_gpu.detach(),
                    "event_times": event_time_cpu,  # Already numpy/list on CPU
                    "hazards_batch": hazards.detach(),  # For patient results
                    "S_batch": S.detach(),  # For patient results
                }
            )
        return output_dict

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        step_output = self._common_step(batch, batch_idx, stage="train")
        main_loss = step_output["main_loss"]

        reg_loss = self._get_regularization_loss()
        total_loss = main_loss + reg_loss

        self.log(
            "train/loss_main",
            main_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        if self.lambda_reg > 0 and self.reg_type:
            self.log(
                "train/loss_reg",
                reg_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        self.log(
            "train/loss_total",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Return loss for PL, and other items needed for epoch_end
        step_output["loss"] = total_loss  # PL needs 'loss' key for backprop
        return step_output

    def _aggregate_epoch_outputs(
        self, outputs: List[Dict[str, Any]], stage: str
    ) -> Dict[str, Any]:
        """Aggregates outputs from all steps in an epoch."""
        agg_data = {
            "avg_loss": torch.stack([x["main_loss"] for x in outputs]).mean().item()
        }  # Avg main loss

        if self.hparams.task_type == "classification":
            all_preds = torch.cat([x["preds"] for x in outputs]).cpu().numpy()
            all_probs = torch.cat([x["probs"] for x in outputs]).cpu().numpy()
            all_labels = torch.cat([x["labels"] for x in outputs]).cpu().numpy()
            agg_data.update(
                {
                    "all_preds": all_preds,
                    "all_probs": all_probs,
                    "all_labels": all_labels,
                }
            )
            if stage == "test" and getattr(
                self.hparams, "collect_patient_results_on_test", False
            ):
                all_logits = (
                    torch.cat([x["logits_batch"] for x in outputs]).cpu().numpy()
                )
                all_slide_ids_flat = [
                    item
                    for sublist in [
                        x["slide_ids"] for x in outputs if x["slide_ids"] is not None
                    ]
                    for item in (sublist if isinstance(sublist, list) else [sublist])
                ]
                agg_data.update(
                    {
                        "all_logits_test": all_logits,
                        "all_slide_ids_test": all_slide_ids_flat,
                    }
                )

        elif self.hparams.task_type == "survival":
            all_risks = torch.cat([x["risks"] for x in outputs]).cpu().numpy()
            all_labels = (
                torch.cat([x["labels"] for x in outputs]).cpu().numpy()
            )  # discrete time bins
            all_censorships = (
                torch.cat([x["censorships"] for x in outputs]).cpu().numpy()
            )
            # event_times are already numpy/list of lists
            all_event_times = (
                np.concatenate([x["event_times"] for x in outputs])
                if outputs and outputs[0]["event_times"] is not None
                else np.array([])
            )

            agg_data.update(
                {
                    "all_risks": all_risks,
                    "all_labels": all_labels,
                    "all_censorships": all_censorships,
                    "all_event_times": all_event_times,
                }
            )
            if stage == "test" and getattr(
                self.hparams, "collect_patient_results_on_test", False
            ):
                all_hazards = (
                    torch.cat([x["hazards_batch"] for x in outputs]).cpu().numpy()
                )
                all_S = torch.cat([x["S_batch"] for x in outputs]).cpu().numpy()
                all_slide_ids_flat = [
                    item
                    for sublist in [
                        x["slide_ids"] for x in outputs if x["slide_ids"] is not None
                    ]
                    for item in (sublist if isinstance(sublist, list) else [sublist])
                ]
                agg_data.update(
                    {
                        "all_hazards_test": all_hazards,
                        "all_S_test": all_S,
                        "all_slide_ids_test": all_slide_ids_flat,
                    }
                )

        return agg_data

    def _log_epoch_metrics(self, agg_data: Dict[str, Any], stage: str):
        """Logs metrics at the end of an epoch."""
        self.log(
            f"{stage}/loss",
            agg_data["avg_loss"],
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if self.hparams.task_type == "classification":
            epoch_error = calculate_error(
                torch.from_numpy(agg_data["all_preds"]),
                torch.from_numpy(agg_data["all_labels"]),
            )
            self.log(
                f"{stage}/error", epoch_error, on_epoch=True, prog_bar=True, logger=True
            )

            if agg_data["all_labels"].size > 0 and agg_data["all_probs"].size > 0:
                epoch_auc = _calculate_classification_auc(
                    agg_data["all_labels"],
                    agg_data["all_probs"],
                    self.hparams.n_classes,
                )
                self.log(
                    f"{stage}/auc", epoch_auc, on_epoch=True, prog_bar=True, logger=True
                )
                self.log(
                    f"{stage}_metric", epoch_auc, on_epoch=True
                )  # For EarlyStopping/ModelCheckpoint if monitoring AUC

                # Per-class accuracy (mimicking AccuracyLogger)
                for i in range(self.hparams.n_classes):
                    class_preds = agg_data["all_preds"][agg_data["all_labels"] == i]
                    class_labels = agg_data["all_labels"][
                        agg_data["all_labels"] == i
                    ]  # Should all be i
                    if len(class_labels) > 0:
                        correct_count = np.sum(
                            class_preds == i
                        )  # Assuming preds are class indices
                        acc = correct_count / len(class_labels)
                        self.log(
                            f"{stage}/class_{i}_acc", acc, on_epoch=True, logger=True
                        )
                        # print(f"    {stage.capitalize()} Class {i}: acc {acc:.4f} ({correct_count}/{len(class_labels)})")

        elif self.hparams.task_type == "survival":
            if len(agg_data["all_event_times"]) > 0:
                epoch_c_index = _calculate_survival_c_index(
                    agg_data["all_event_times"],
                    agg_data["all_censorships"],
                    agg_data["all_risks"],
                )
                self.log(
                    f"{stage}/c_index",
                    epoch_c_index,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
                self.log(
                    f"{stage}_metric", epoch_c_index, on_epoch=True
                )  # For EarlyStopping/ModelCheckpoint
            else:
                self.log(
                    f"{stage}/c_index",
                    np.nan,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
                self.log(f"{stage}_metric", np.nan, on_epoch=True)

    def training_epoch_end(self, outputs: List[Dict[str, Any]]):
        agg_data = self._aggregate_epoch_outputs(outputs, stage="train")
        self._log_epoch_metrics(agg_data, stage="train")
        # Log learning rate
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, on_epoch=True, logger=True)

    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict[str, Any]:
        step_output = self._common_step(batch, batch_idx, stage="val")
        self.log(
            "val/loss_step",
            step_output["main_loss"],
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        return step_output

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]):
        agg_data = self._aggregate_epoch_outputs(outputs, stage="val")
        self._log_epoch_metrics(agg_data, stage="val")
        # PL uses 'val_loss' by default for some callbacks, or you can specify 'val_metric'
        self.log(
            "val_loss", agg_data["avg_loss"], on_epoch=True, prog_bar=True
        )  # Ensure val_loss is logged for schedulers/callbacks

    def test_step(self, batch: Tuple, batch_idx: int) -> Dict[str, Any]:
        step_output = self._common_step(batch, batch_idx, stage="test")
        if (
            getattr(self.hparams, "collect_patient_results_on_test", False)
            and step_output["slide_ids"] is not None
        ):
            # This collects raw outputs per batch. Aggregation into dict happens in test_epoch_end
            self.test_patient_results.append(step_output)
        return step_output

    def test_epoch_end(self, outputs: List[Dict[str, Any]]):
        agg_data = self._aggregate_epoch_outputs(outputs, stage="test")
        self._log_epoch_metrics(agg_data, stage="test")

        if (
            getattr(self.hparams, "collect_patient_results_on_test", False)
            and self.test_patient_results
        ):
            # Aggregate collected patient results into the desired dictionary format
            # This is a simplified aggregation; adjust as needed for the exact structure.
            self.test_patient_results_aggregated = {}
            num_samples = len(agg_data.get("all_slide_ids_test", []))

            for i in range(num_samples):
                slide_id = str(agg_data["all_slide_ids_test"][i])
                if self.hparams.task_type == "classification":
                    self.test_patient_results_aggregated[slide_id] = {
                        "slide_id": slide_id,
                        "prob": agg_data["all_probs"][i].tolist(),
                        "pred": int(agg_data["all_preds"][i]),
                        "label": int(agg_data["all_labels"][i]),
                        "logits": agg_data["all_logits_test"][i].tolist(),
                    }
                elif self.hparams.task_type == "survival":
                    self.test_patient_results_aggregated[slide_id] = {
                        "slide_id": slide_id,
                        "risk": float(agg_data["all_risks"][i]),
                        "disc_label": int(
                            agg_data["all_labels"][i]
                        ),  # discrete time bin label
                        "survival_time": float(
                            agg_data["all_event_times"][i]
                        ),  # actual event time
                        "censorship": int(agg_data["all_censorships"][i]),
                        "hazards": agg_data["all_hazards_test"][i].tolist(),
                        "S": agg_data["all_S_test"][i].tolist(),
                    }
            # Clear the per-batch list after aggregation
            self.test_patient_results = []
            print(
                f"Aggregated {len(self.test_patient_results_aggregated)} patient results in test_epoch_end."
            )

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Tuple[List, List]]:
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        optimizer = get_optim(
            self.model, self.hparams
        )  # Assuming get_optim takes model and hparams (args)

        if (
            hasattr(self.hparams, "lr_scheduler_name")
            and self.hparams.lr_scheduler_name
        ):
            if self.hparams.lr_scheduler_name.lower() == "plateau":
                scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode=getattr(
                        self.hparams, "lr_scheduler_mode", "min"
                    ),  # 'min' for val_loss, 'max' for val_metric
                    factor=getattr(self.hparams, "lr_scheduler_factor", 0.5),
                    patience=getattr(self.hparams, "lr_scheduler_patience", 10),
                    verbose=True,
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss"
                        if getattr(self.hparams, "lr_scheduler_mode", "min") == "min"
                        else "val_metric",  # or "val_metric"
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            # Add other schedulers like CosineAnnealingLR here
            # elif self.hparams.lr_scheduler_name.lower() == 'cosine':
            #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs) # Ensure max_epochs is in hparams
            #     return [optimizer], [scheduler]

        return optimizer
