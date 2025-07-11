# sauron/training/lightning_module.py

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from sksurv.metrics import concordance_index_censored
from torch.optim.lr_scheduler import CosineAnnealingLR

from sauron.losses.surv_loss import CoxSurvLoss, NLLSurvLoss
from sauron.mil_models.models_factory import mil_model_factory
from sauron.utils.optimizers import get_optim


class Sauron(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)  # Save args to checkpoint
        self.model = mil_model_factory(args)

        # Loss function
        if self.args.task_type.lower() == "classification":
            self.loss_fn = nn.CrossEntropyLoss()
            # Metrics for classification
            self.train_auc = torchmetrics.AUROC(
                task="multiclass", num_classes=args.n_classes
            )
            self.val_auc = torchmetrics.AUROC(
                task="multiclass", num_classes=args.n_classes
            )
            self.test_auc = torchmetrics.AUROC(
                task="multiclass", num_classes=args.n_classes
            )
            self.train_acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=args.n_classes
            )
            self.val_acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=args.n_classes
            )
            self.test_acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=args.n_classes
            )
        elif self.args.task_type.lower() == "survival":
            if self.args.bag_loss == "nll_surv":
                self.loss_fn = NLLSurvLoss(alpha=self.args.alpha_surv)
            elif self.args.bag_loss == "cox_surv":
                self.loss_fn = CoxSurvLoss()
            else:
                raise ValueError(f"Unknown survival loss: {self.args.bag_loss}")
        else:
            raise ValueError(f"Unknown task type: {self.args.task_type}")

    def forward(self, x):
        return self.model(x)

    def _get_outputs_and_loss(self, batch):
        data, label, event, c = batch  # Assuming dataloader provides these

        if self.args.task_type.lower() == "classification":
            logits, probs, preds, _, _ = self.model(data)
            loss = self.loss_fn(logits, label)
            return loss, logits, probs, label, None, None

        elif self.args.task_type.lower() == "survival":
            hazards, S, preds, _, _ = self.model(data)
            loss = self.loss_fn(hazards=hazards, S=S, Y=label, c=c)
            risk = -torch.sum(S, dim=1)
            return loss, risk, None, event, c, None

    def training_step(self, batch, batch_idx):
        loss, outputs, _, _, _, _ = self._get_outputs_and_loss(batch)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch[0].size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs, probs, labels, c, _ = self._get_outputs_and_loss(batch)
        self.log(
            "val_loss", loss, prog_bar=True, logger=True, batch_size=batch[0].size(0)
        )

        if self.args.task_type.lower() == "classification":
            self.val_auc.update(probs, labels)
            self.val_acc.update(probs, labels)
            self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        elif self.args.task_type.lower() == "survival":
            # For survival, we need to collect all outputs to compute c-index at the end of epoch
            return {"risk": outputs, "event": labels, "c": c}

    def on_validation_epoch_end(self):
        if self.args.task_type.lower() == "classification":
            self.log("val_auc", self.val_auc.compute(), prog_bar=True)
            self.val_auc.reset()
            self.val_acc.reset()
        elif self.args.task_type.lower() == "survival":
            # This part is tricky with PyTorch Lightning; a common approach is to collect outputs
            # from validation_step and compute the metric here.
            # As of recent versions, this requires manual collection.
            # A simpler approach for now is to compute it in a callback or outside the main loop
            # based on the best checkpoint.
            # For a complete PL implementation, you'd collect outputs from `validation_step`.
            pass

    def test_step(self, batch, batch_idx):
        loss, outputs, probs, labels, c, _ = self._get_outputs_and_loss(batch)
        self.log("test_loss", loss, logger=True, batch_size=batch[0].size(0))

        if self.args.task_type.lower() == "classification":
            self.test_auc.update(probs, labels)
            self.test_acc.update(probs, labels)
        elif self.args.task_type.lower() == "survival":
            return {
                "risk": outputs.cpu().numpy(),
                "event": labels.cpu().numpy(),
                "c": c.cpu().numpy(),
            }

    def on_test_epoch_end(self, outputs):
        if self.args.task_type.lower() == "classification":
            self.log("test_auc", self.test_auc.compute())
            self.log("test_acc", self.test_acc.compute())
        elif self.args.task_type.lower() == "survival" and outputs:
            risks = np.concatenate([o["risk"] for o in outputs])
            events = np.concatenate([o["event"] for o in outputs])
            cs = np.concatenate([o["c"] for o in outputs])
            event_observed = (1 - cs).astype(bool)
            c_index = concordance_index_censored(event_observed, events, risks)[0]
            self.log("test_c_index", c_index)

    def configure_optimizers(self):
        optimizer = get_optim(self.model, self.args)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.max_epochs)
        return [optimizer], [scheduler]
