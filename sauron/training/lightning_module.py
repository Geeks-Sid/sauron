import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR

from sauron.mil_models.models_factory import mil_model_factory
from sauron.losses.surv_loss import NLLSurvLoss, CoxPHSurvLoss
from sauron.utils.metrics import concordance_index, accuracy_cox

class Sauron(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = mil_model_factory(args)
        if self.args.task == 'classification':
            self.loss = nn.CrossEntropyLoss()
        elif self.args.task == 'survival':
            if self.args.loss == 'nll':
                self.loss = NLLSurvLoss()
            elif self.args.loss == 'cox':
                self.loss = CoxPHSurvLoss()
            else:
                raise ValueError(f"Unknown survival loss: {self.args.loss}")
        else:
            raise ValueError(f"Unknown task: {self.args.task}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, label, event = batch
        output = self(data)
        loss = self.loss(output, label, event)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label, event = batch
        output = self(data)
        loss = self.loss(output, label, event)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.args.task == 'survival':
            c_index = concordance_index(output, label, event)
            self.log('c_index', c_index, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        return [optimizer], [scheduler]