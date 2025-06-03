# utils/training_utils.py

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,  # roc_curve was unused, calc_auc was roc_auc_score
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sauron.losses.surv_loss import (
    CoxSurvLoss,
    CrossEntropySurvLoss,
    NLLSurvLoss,
)

# --- Utility Imports ---
from sauron.utils.callbacks import (
    AccuracyLogger,
    EarlyStopping,
)
from sauron.utils.generic_utils import calculate_error
from sauron.utils.metrics import (
    _calculate_classification_auc,
    _calculate_survival_c_index,
)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        task_type: str,
        n_classes: int,
        args: Any,
        writer: Optional[SummaryWriter] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.task_type = task_type
        self.n_classes = n_classes
        self.writer = writer
        self.args = args

        self.grad_accum_steps = getattr(args, "gc", 1)
        self.lambda_reg = getattr(args, "lambda_reg", 0.0)
        self.reg_fn = None  # Define if L1/L2 reg on weights is needed
        # Example L1 reg_fn:
        # if self.lambda_reg > 0 and getattr(args, "reg_type", None) == "l1":
        #     self.reg_fn = lambda m: sum(p.abs().sum() for p in m.parameters() if p.requires_grad)

        self.train_survival_loss_alpha = (
            args.alpha_surv
            if task_type == "survival" and hasattr(args, "alpha_surv")
            else 0.0
        )

        self.scheduler = None
        if hasattr(args, "lr_scheduler_name") and args.lr_scheduler_name:
            if args.lr_scheduler_name.lower() == "plateau":
                self.scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode=getattr(
                        args, "lr_scheduler_mode", "min"
                    ),  # 'min' for loss, 'max' for metric
                    factor=getattr(args, "lr_scheduler_factor", 0.5),
                    patience=getattr(args, "lr_scheduler_patience", 10),
                )
            # elif args.lr_scheduler_name.lower() == 'cosine':
            #     self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
            # Add other schedulers here

        self.current_epoch = 0
        self.train_loader: Optional[DataLoader] = None  # Will be set in fit()

    def _unpack_batch_data(
        self, batch_data: Tuple
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[Any], Optional[torch.Tensor], Optional[Any]
    ]:
        """Unpacks batch data. Returns (model_input, label, event_time_cpu, censorship_gpu, slide_id (if available))."""
        slide_id = None  # Default
        if self.task_type == "classification":
            if len(batch_data) == 2:
                data, label = batch_data
            elif len(batch_data) == 3:
                data, label, slide_id = batch_data
            else:
                raise ValueError(
                    f"Unexpected batch data length for classification: {len(batch_data)}"
                )

            model_input = data.to(self.device, non_blocking=True)
            label_tensor = label.to(self.device, non_blocking=True)
            return model_input, label_tensor, None, None, slide_id

        elif self.task_type == "survival":
            # Original expected (img_tensor, omic_tensor, label, event_time, c)
            # Assuming omic_tensor might be at index 1 and ignored.
            # slide_id could be the 6th element if present.
            if len(batch_data) == 5:
                data_WSI, _omic, label, event_time, c = batch_data
            elif len(batch_data) == 6:
                data_WSI, _omic, label, event_time, c, slide_id = batch_data
            else:
                raise ValueError(
                    f"Unexpected batch data length for survival: {len(batch_data)}"
                )

            model_input = data_WSI.to(self.device, non_blocking=True)
            label_tensor = label.to(
                self.device, non_blocking=True
            )  # Discrete time bin label
            c_tensor = c.to(self.device, non_blocking=True)
            # event_time is often kept on CPU as it's used with numpy for C-index.
            # It's not directly used in loss for many survival models unless transformed.
            event_time_cpu = (
                event_time.cpu().numpy()
                if isinstance(event_time, torch.Tensor)
                else event_time
            )
            return model_input, label_tensor, event_time_cpu, c_tensor, slide_id
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

    def _calculate_main_loss(
        self,
        model_outputs: Tuple,
        label: torch.Tensor,
        c_tensor: Optional[torch.Tensor],
        current_survival_alpha: float,
    ) -> torch.Tensor:
        if self.task_type == "classification":
            logits, _, _, _, _ = model_outputs
            return self.loss_fn(logits, label)
        elif self.task_type == "survival":
            hazards, S, _, _, _ = model_outputs
            return self.loss_fn(
                hazards=hazards, S=S, Y=label, c=c_tensor, alpha=current_survival_alpha
            )
        raise ValueError(f"Unknown task_type for loss calculation: {self.task_type}")

    def _train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        acc_logger = (
            AccuracyLogger(n_classes=self.n_classes)
            if self.task_type == "classification"
            else None
        )

        epoch_main_loss = 0.0
        epoch_total_loss = 0.0

        risk_scores_epoch, censorships_epoch, event_times_epoch = [], [], []

        self.optimizer.zero_grad()
        assert self.train_loader is not None, "Train loader not set in Trainer"

        with tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch} [Train]",
            leave=False,
            ncols=100,
        ) as pbar:
            for batch_idx, batch_data in enumerate(pbar):
                model_input, label_gpu, event_time_cpu, c_gpu, _ = (
                    self._unpack_batch_data(batch_data)
                )

                model_outputs = self.model(
                    model_input
                )  # AMP: with torch.cuda.amp.autocast():
                main_loss = self._calculate_main_loss(
                    model_outputs, label_gpu, c_gpu, self.train_survival_loss_alpha
                )

                reg_loss_item = 0.0
                if self.reg_fn and self.lambda_reg > 0:
                    reg_loss_val = self.reg_fn(self.model) * self.lambda_reg
                    reg_loss_item = reg_loss_val.item()
                    combined_loss = main_loss + reg_loss_val
                else:
                    combined_loss = main_loss

                loss_scaled = combined_loss / self.grad_accum_steps
                loss_scaled.backward()  # AMP: scaler.scale(loss_scaled).backward()

                epoch_main_loss += main_loss.item()
                epoch_total_loss += combined_loss.item()

                postfix_dict = {"loss": f"{main_loss.item():.4f}"}
                if self.lambda_reg > 0 and reg_loss_item > 0:
                    postfix_dict["reg"] = f"{reg_loss_item:.4f}"

                if self.task_type == "classification":
                    _, _, y_hat, _, _ = model_outputs
                    error = calculate_error(y_hat, label_gpu)
                    acc_logger.log(y_hat, label_gpu)
                    postfix_dict["err"] = f"{error:.4f}"
                elif self.task_type == "survival":
                    _, S, _, _, _ = model_outputs
                    risk = -torch.sum(S, dim=1).detach().cpu().numpy()
                    risk_scores_epoch.extend(risk)
                    censorships_epoch.extend(
                        c_gpu.cpu().numpy()
                    )  # c_gpu already on device
                    event_times_epoch.extend(
                        event_time_cpu
                    )  # event_time_cpu is already numpy array or list

                pbar.set_postfix(**postfix_dict)

                if (batch_idx + 1) % self.grad_accum_steps == 0 or (
                    batch_idx + 1
                ) == len(self.train_loader):
                    # AMP: scaler.unscale_(optimizer)
                    # if getattr(self.args, "clip_grad_norm", None):
                    #    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_grad_norm)
                    self.optimizer.step()  # AMP: scaler.step(optimizer)
                    # AMP: scaler.update()
                    self.optimizer.zero_grad()

        avg_main_loss = epoch_main_loss / len(self.train_loader)
        avg_total_loss = epoch_total_loss / len(self.train_loader)

        print(
            f"Epoch {self.current_epoch} [Train]: AvgLoss(Main): {avg_main_loss:.4f}, AvgLoss(Total): {avg_total_loss:.4f}",
            end="",
        )

        train_metric_val = np.nan  # Default to NaN
        if self.task_type == "classification":
            assert acc_logger is not None
            train_error = acc_logger.get_average_error()
            train_metric_val = train_error  # For classification, error is a common metric (lower is better)
            print(f", AvgError: {train_error:.4f}")
            if self.writer:
                self.writer.add_scalar("train/error", train_error, self.current_epoch)
                for i in range(self.n_classes):
                    acc, correct, count = acc_logger.get_summary(i)
                    print(f"    Class {i}: acc {acc:.4f} ({correct}/{count})", end=" ")
                    self.writer.add_scalar(
                        f"train/class_{i}_acc", acc, self.current_epoch
                    )
                print()  # Newline after class accuracies
        elif self.task_type == "survival":
            c_index = _calculate_survival_c_index(
                np.array(event_times_epoch),
                np.array(censorships_epoch),
                np.array(risk_scores_epoch),
            )
            train_metric_val = c_index  # Higher is better
            print(f", C-Index: {c_index:.4f}")
            if self.writer:
                self.writer.add_scalar("train/c_index", c_index, self.current_epoch)

        if self.writer:
            self.writer.add_scalar("train/loss_main", avg_main_loss, self.current_epoch)
            self.writer.add_scalar(
                "train/loss_total", avg_total_loss, self.current_epoch
            )
            self.writer.add_scalar(
                "train/lr", self.optimizer.param_groups[0]["lr"], self.current_epoch
            )
        return avg_main_loss, train_metric_val

    def _evaluate(
        self,
        loader: DataLoader,
        epoch_log_id: int,
        eval_survival_loss_alpha: float = 0.0,
        results_prefix: str = "val",
        collect_patient_results: bool = False,
    ) -> Tuple[float, float, Optional[Dict[str, Any]]]:
        self.model.eval()
        acc_logger = (
            AccuracyLogger(n_classes=self.n_classes)
            if self.task_type == "classification"
            else None
        )

        epoch_loss = 0.0
        all_probs_or_risks, all_labels_cpu, all_censorships_cpu, all_event_times_cpu = (
            [],
            [],
            [],
            [],
        )

        patient_results_dict: Dict[str, Any] = {}
        # Slide ID collection depends on batch_size=1 for simple indexing if not provided by loader
        # It's safer if loader yields slide_ids.
        can_get_slide_id_from_dataset = (
            hasattr(loader.dataset, "slide_data")
            and "slide_id" in loader.dataset.slide_data.columns
            and loader.batch_size == 1
        )  # Only reliable if bs=1 and not shuffled

        with torch.no_grad(), tqdm(
            loader,
            desc=f"Epoch {self.current_epoch} [{results_prefix.upper()}]",
            leave=False,
            ncols=100,
        ) as pbar:
            for batch_idx, batch_data in enumerate(pbar):
                model_input, label_gpu, event_time_b_cpu, c_gpu, slide_id_b = (
                    self._unpack_batch_data(batch_data)
                )

                current_slide_id_str = None
                if collect_patient_results:
                    if (
                        slide_id_b is not None
                    ):  # slide_id provided by dataloader for the item
                        current_slide_id_str = str(
                            slide_id_b[0]
                            if isinstance(slide_id_b, (list, tuple, torch.Tensor))
                            and len(slide_id_b) > 0
                            else slide_id_b
                        )
                    elif can_get_slide_id_from_dataset:  # Fallback for bs=1
                        current_slide_id_str = str(
                            loader.dataset.slide_data["slide_id"].iloc[batch_idx]
                        )

                model_outputs = self.model(model_input)
                loss = self._calculate_main_loss(
                    model_outputs, label_gpu, c_gpu, eval_survival_loss_alpha
                )
                epoch_loss += loss.item()

                if self.task_type == "classification":
                    logits, y_prob, y_hat, _, _ = model_outputs
                    assert acc_logger is not None
                    acc_logger.log_batch(
                        y_hat.cpu().numpy(), label_gpu.cpu().numpy()
                    )  # log_batch for full batch
                    all_probs_or_risks.append(y_prob.cpu().numpy())
                    all_labels_cpu.append(label_gpu.cpu().numpy())
                    if collect_patient_results and current_slide_id_str is not None:
                        patient_results_dict[current_slide_id_str] = {
                            "slide_id": current_slide_id_str,
                            "prob": y_prob.cpu().numpy()[0].tolist(),  # bs=1
                            "pred": y_hat.item(),
                            "label": label_gpu.item(),  # bs=1
                            "logits": logits.cpu().numpy()[0].tolist(),  # bs=1
                        }
                elif self.task_type == "survival":
                    hazards, S, _, _, _ = model_outputs
                    risk = -torch.sum(S, dim=1).cpu().numpy()  # S is [B, N_intervals]
                    all_probs_or_risks.extend(risk)  # risk is already [B] numpy array
                    all_labels_cpu.append(
                        label_gpu.cpu().numpy()
                    )  # label is [B] discrete time bin
                    all_censorships_cpu.extend(c_gpu.cpu().numpy())  # c_gpu is [B]
                    all_event_times_cpu.extend(
                        event_time_b_cpu
                    )  # event_time_b_cpu is [B] numpy array
                    if collect_patient_results and current_slide_id_str is not None:
                        patient_results_dict[
                            current_slide_id_str
                        ] = {  # Assuming bs=1 for simplicity here
                            "slide_id": current_slide_id_str,
                            "risk": risk[0],
                            "disc_label": label_gpu.item(),
                            "survival": event_time_b_cpu.item()
                            if isinstance(event_time_b_cpu, np.ndarray)
                            and event_time_b_cpu.size == 1
                            else event_time_b_cpu[0],
                            "censorship": c_gpu.cpu().numpy().item(),
                            "hazards": hazards.cpu().numpy()[0].tolist(),
                            "S": S.cpu().numpy()[0].tolist(),
                        }

        avg_loss = epoch_loss / len(loader)
        metric_val = np.nan

        print_prefix = f"Epoch {self.current_epoch} [{results_prefix.upper()}]: AvgLoss: {avg_loss:.4f}"

        if self.task_type == "classification":
            all_probs_np = (
                np.concatenate(all_probs_or_risks)
                if len(all_probs_or_risks) > 0
                else np.array([])
            )
            all_labels_np = (
                np.concatenate(all_labels_cpu)
                if len(all_labels_cpu) > 0
                else np.array([])
            )

            if all_labels_np.size > 0 and all_probs_np.size > 0:
                metric_val = _calculate_classification_auc(
                    all_labels_np, all_probs_np, self.n_classes
                )
                eval_error = acc_logger.get_average_error()
                print(
                    f"{print_prefix}, AvgError: {eval_error:.4f}, AUC: {metric_val:.4f}"
                )
                if self.writer:
                    self.writer.add_scalar(
                        f"{results_prefix}/error", eval_error, epoch_log_id
                    )
                    self.writer.add_scalar(
                        f"{results_prefix}/auc", metric_val, epoch_log_id
                    )
                    for i in range(self.n_classes):
                        acc, correct, count = acc_logger.get_summary(i)
                        print(
                            f"    Class {i}: acc {acc:.4f} ({correct}/{count})", end=" "
                        )
                        self.writer.add_scalar(
                            f"{results_prefix}/class_{i}_acc", acc, epoch_log_id
                        )
                    print()  # Newline
            else:
                print(f"{print_prefix}, No data for metrics.")

        elif self.task_type == "survival":
            if len(all_event_times_cpu) > 0:
                metric_val = _calculate_survival_c_index(
                    np.array(all_event_times_cpu),
                    np.array(all_censorships_cpu),
                    np.array(all_probs_or_risks),
                )
                print(f"{print_prefix}, C-Index: {metric_val:.4f}")
                if self.writer:
                    self.writer.add_scalar(
                        f"{results_prefix}/c_index", metric_val, epoch_log_id
                    )
            else:
                print(f"{print_prefix}, No data for metrics.")

        if self.writer:
            self.writer.add_scalar(f"{results_prefix}/loss", avg_loss, epoch_log_id)

        return (
            metric_val,
            avg_loss,
            (
                patient_results_dict
                if collect_patient_results and patient_results_dict
                else None
            ),
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        early_stopping_cb: Optional[EarlyStopping] = None,
        max_epochs: int = 100,
    ):
        self.train_loader = train_loader

        for epoch in range(max_epochs):
            self.current_epoch = epoch
            start_time = time.time()

            _, _ = self._train_epoch()  # train_loss, train_metric (already logged)

            val_survival_alpha = (
                0.0 if self.task_type == "survival" else self.train_survival_loss_alpha
            )  # Use 0 alpha for val loss in survival
            val_metric, val_loss, _ = self._evaluate(
                val_loader,
                epoch_log_id=self.current_epoch,
                eval_survival_loss_alpha=val_survival_alpha,
                results_prefix="val",
            )

            print(f"Epoch {self.current_epoch} Time: {time.time() - start_time:.2f}s")

            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    # Plateau scheduler steps on val_loss or val_metric (needs to match EarlyStopping mode for consistency)
                    plateau_monitor_val = (
                        val_loss if self.scheduler.mode == "min" else val_metric
                    )
                    self.scheduler.step(plateau_monitor_val)
                else:  # For schedulers like CosineAnnealingLR
                    self.scheduler.step()

            if early_stopping_cb:
                # EarlyStopping callback should use 'metric' or 'loss' for its stop_metric attribute
                monitor_val_for_es = (
                    val_loss if early_stopping_cb.stop_metric == "loss" else val_metric
                )
                early_stopping_cb(epoch, monitor_val_for_es, self.model)
                if early_stopping_cb.early_stop:
                    print(f"Early stopping triggered at epoch {self.current_epoch}.")
                    break

        # After loop, load best model if early stopping was used and saved a model
        if (
            early_stopping_cb
            and hasattr(early_stopping_cb, "ckpt_path")
            and early_stopping_cb.ckpt_path
            and os.path.exists(early_stopping_cb.ckpt_path)
        ):
            print(
                f"Loading best model from early stopping checkpoint: {early_stopping_cb.ckpt_path}"
            )
            try:
                self.model.load_state_dict(
                    torch.load(early_stopping_cb.ckpt_path, map_location=self.device)
                )
            except Exception as e:
                print(
                    f"Error loading state dict from {early_stopping_cb.ckpt_path}: {e}. Using last model state."
                )
        elif early_stopping_cb:
            print(
                f"Warning: Early stopping was enabled, but best model checkpoint '{getattr(early_stopping_cb, 'ckpt_path', 'N/A')}' not found or not saved. Using last model state."
            )
        else:
            print(
                "Training finished (max_epochs reached or no early stopping). Using last model state."
            )
