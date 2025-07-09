import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from sauron.data.dataset_factory import dataset_factory
from sauron.training.lightning_module import Sauron

def run_pipeline(args):
    train_dataset = dataset_factory(args, 'train')
    val_dataset = dataset_factory(args, 'val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = Sauron(args)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/{args.experiment_name}',
        filename='{epoch}-{val_loss:.2f}-{c_index:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode='min'
    )

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    trainer.fit(model, train_loader, val_loader)