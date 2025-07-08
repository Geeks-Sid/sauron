import os
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import Logger, TensorBoardLogger
from torch.utils.data import DataLoader

from sauron.data.data_utils import get_dataloader
from sauron.mil_models.models_factory import initialize_mil_model
from sauron.training.lightning_module import SauronLightningModule

# --- Constants for default values (if not in args) ---
DEFAULT_NUM_WORKERS = 4
DEFAULT_VAL_TEST_BATCH_SIZE = 1
DEFAULT_ES_PATIENCE = 20
DEFAULT_GRAD_ACCUM_STEPS = 1
DEFAULT_LOG_EVERY_N_STEPS = 50
DEFAULT_TQDM_REFRESH_DIVISOR = 100


def _setup_environment_and_logger(
    base_results_dir: str, current_fold_num: int, log_data_flag: bool
) -> Tuple[str, Optional[Logger]]:
    """Sets up the results directory for the current fold and initializes the TensorBoardLogger."""
    fold_results_dir = os.path.join(base_results_dir, str(current_fold_num))
    os.makedirs(fold_results_dir, exist_ok=True)

    logger = None
    if log_data_flag:
        logger = TensorBoardLogger(
            save_dir=base_results_dir,
            name="",  # No subdirectory for model name, version handles fold
            version=str(current_fold_num),
            default_hp_metric=False,  # Recommended by PyTorch Lightning
        )
    print(f"Results directory for fold {current_fold_num}: {fold_results_dir}")
    return fold_results_dir, logger


def _initialize_dataloaders(
    train_dataset: Any,
    val_dataset: Any,
    test_dataset: Optional[Any],
    args: Any,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Initializes train, validation, and optional test DataLoaders."""
    val_test_batch_size = getattr(
        args, "val_test_batch_size", DEFAULT_VAL_TEST_BATCH_SIZE
    )
    num_workers = getattr(args, "num_workers", DEFAULT_NUM_WORKERS)

    train_loader = get_dataloader(
        train_dataset,
        training=True,
        weighted=args.weighted_sample,
        batch_size=args.batch_size,
        num_workers=num_workers,
    )
    val_loader = get_dataloader(
        val_dataset,
        training=False,
        batch_size=val_test_batch_size,
        num_workers=num_workers,
    )
    test_loader = (
        get_dataloader(
            test_dataset,
            training=False,
            batch_size=val_test_batch_size,
            num_workers=num_workers,
        )
        if test_dataset
        else None
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    if test_dataset:
        print(f"Test dataset size: {len(test_dataset)}")
    return train_loader, val_loader, test_loader


def _initialize_lightning_module(args: Any) -> SauronLightningModule:
    """Initializes the PyTorch model and wraps it with SauronLightningModule."""
    pytorch_model = initialize_mil_model(args)
    lightning_module = SauronLightningModule(model=pytorch_model, args=args)
    return lightning_module


def _configure_callbacks(
    args: Any, fold_results_dir: str, train_loader_len: int
) -> Tuple[List[Callback], ModelCheckpoint]:
    """Configures and returns PyTorch Lightning callbacks."""
    callbacks_list: List[Callback] = []

    # Determine monitor metric and mode for EarlyStopping and ModelCheckpoint
    monitor_metric_name = "val_loss"
    monitor_mode = "min"
    filename_template = "best_model-{epoch}-{val_loss:.4f}"

    if args.task_type == "classification":
        if (
            getattr(args, "monitor_metric", "loss").lower() == "metric"
        ):  # 'metric' typically AUC for classification
            monitor_metric_name = "val/auc"
            monitor_mode = "max"
            filename_template = "best_model-{epoch}-{val/auc:.4f}"
    elif args.task_type == "survival":  # For survival, 'metric' is C-Index
        if getattr(args, "monitor_metric", "loss").lower() == "metric":
            monitor_metric_name = "val/c_index"
            monitor_mode = "max"
            filename_template = "best_model-{epoch}-{val/c_index:.4f}"
        # If monitor_metric is 'loss' for survival, default val_loss and min mode are already set

    if args.early_stopping:
        early_stopping_callback = EarlyStopping(
            monitor=monitor_metric_name,
            patience=getattr(args, "es_patience", DEFAULT_ES_PATIENCE),
            verbose=True,
            mode=monitor_mode,
        )
        callbacks_list.append(early_stopping_callback)
        print(
            f"Early stopping enabled: monitor='{monitor_metric_name}', mode='{monitor_mode}'"
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(fold_results_dir, "checkpoints"),
        filename=filename_template,
        monitor=monitor_metric_name,
        mode=monitor_mode,
        save_top_k=1,
        save_last=True,
        verbose=True,
    )
    callbacks_list.append(checkpoint_callback)
    print(
        f"Model checkpointing enabled: monitor='{monitor_metric_name}', mode='{monitor_mode}'"
    )

    # TQDM Progress Bar
    refresh_rate = max(
        1,
        train_loader_len // DEFAULT_TQDM_REFRESH_DIVISOR if train_loader_len > 0 else 1,
    )
    callbacks_list.append(TQDMProgressBar(refresh_rate=refresh_rate))

    return callbacks_list, checkpoint_callback


def _configure_trainer(
    args: Any,
    logger: Optional[Logger],
    callbacks: List[Callback],
    train_loader_len: int,
) -> pl.Trainer:
    """Configures and returns the PyTorch Lightning Trainer."""
    grad_accum_steps = getattr(args, "gc", DEFAULT_GRAD_ACCUM_STEPS)

    trainer_params = {
        "logger": logger,
        "callbacks": callbacks,
        "max_epochs": args.max_epochs,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": "auto"
        if torch.cuda.is_available()
        else 1,  # Simpler device selection
        "accumulate_grad_batches": grad_accum_steps,
        "deterministic": getattr(args, "deterministic", False),
        "log_every_n_steps": min(
            DEFAULT_LOG_EVERY_N_STEPS,
            train_loader_len // grad_accum_steps
            if train_loader_len > 0 and grad_accum_steps > 0
            else DEFAULT_LOG_EVERY_N_STEPS,
        ),
        # "precision": "16-mixed" if getattr(args, "amp", False) else 32, # Updated precision flag
    }
    if getattr(args, "amp", False):  # Automatic Mixed Precision
        trainer_params["precision"] = "16-mixed"

    if hasattr(args, "gradient_clip_val") and args.gradient_clip_val is not None:
        trainer_params["gradient_clip_val"] = args.gradient_clip_val
        trainer_params["gradient_clip_algorithm"] = getattr(
            args, "gradient_clip_algorithm", "norm"
        )

    return pl.Trainer(**trainer_params)


def _run_final_evaluation(
    trainer: pl.Trainer,
    lightning_module: SauronLightningModule,
    test_loader: Optional[DataLoader],
    checkpoint_callback: ModelCheckpoint,
    fold_results_dir: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Runs final evaluation on the test set using the best model checkpoint."""
    test_metrics = None
    patient_results_dict = None

    best_model_path = checkpoint_callback.best_model_path

    # Fallback to last checkpoint if best_model_path is empty or doesn't exist
    if not best_model_path or not os.path.exists(best_model_path):
        if best_model_path:  # It was set but file is missing
            print(f"Warning: Best model path '{best_model_path}' does not exist.")

        last_ckpt_path = os.path.join(fold_results_dir, "checkpoints", "last.ckpt")
        if os.path.exists(last_ckpt_path):
            print(f"Attempting to use last checkpoint: '{last_ckpt_path}'")
            best_model_path = last_ckpt_path
        else:
            print(
                "Warning: No best or last checkpoint found. Skipping test evaluation."
            )
            return test_metrics, patient_results_dict

    if test_loader:
        print(f"\n--- Final Test Evaluation (using model from: {best_model_path}) ---")
        trainer.test(
            model=lightning_module, dataloaders=test_loader, ckpt_path=best_model_path
        )

        current_test_metrics = {}
        for key, value in trainer.callback_metrics.items():
            if key.startswith("test/"):
                metric_name = key.replace("test/", "")
                current_test_metrics[metric_name] = (
                    value.item() if isinstance(value, torch.Tensor) else value
                )
        test_metrics = current_test_metrics

        if (
            hasattr(lightning_module, "test_patient_results_aggregated")
            and lightning_module.test_patient_results_aggregated
        ):
            patient_results_dict = lightning_module.test_patient_results_aggregated
            # Optionally save patient results to a file here if desired
            # import json
            # patient_results_path = os.path.join(fold_results_dir, "test_patient_results.json")
            # with open(patient_results_path, 'w') as f:
            #     json.dump(patient_results_dict, f, indent=4)
            # print(f"Saved patient-level test results to {patient_results_path}")
    else:
        print("No test loader provided. Skipping final test evaluation.")

    return test_metrics, patient_results_dict


def _compile_results(
    current_fold_num: int,
    args: Any,
    checkpoint_callback: ModelCheckpoint,
    trainer: pl.Trainer,
    test_metrics: Optional[Dict[str, Any]],
    patient_results_dict: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compiles all training and evaluation results into a dictionary."""
    results = {"fold": current_fold_num}

    if test_metrics:
        for metric_name, value in test_metrics.items():
            results[f"test_{metric_name}"] = value  # e.g. test_auc, test_loss

    if patient_results_dict:
        results["test_patient_results_dict"] = patient_results_dict

    # Add best validation metric from the checkpoint callback
    if checkpoint_callback.best_model_score is not None:
        # Sanitize monitor name for use as a key
        monitor_key = checkpoint_callback.monitor.replace("/", "_").replace("-", "_")
        results[f"best_val_{monitor_key}"] = (
            checkpoint_callback.best_model_score.item()
            if isinstance(checkpoint_callback.best_model_score, torch.Tensor)
            else checkpoint_callback.best_model_score
        )

    # For k-fold summary, the primary monitored validation metric is often used
    if args.k_fold:
        if checkpoint_callback.best_model_score is not None:
            results["kfold_val_metric"] = (
                checkpoint_callback.best_model_score.item()
                if isinstance(checkpoint_callback.best_model_score, torch.Tensor)
                else checkpoint_callback.best_model_score
            )
        else:
            # Fallback if best_model_score is somehow None (e.g., training interrupted before first validation)
            results["kfold_val_metric"] = trainer.callback_metrics.get(
                checkpoint_callback.monitor, float("nan")
            )

    return results


def train_fold(
    train_dataset: Any,
    val_dataset: Any,
    test_dataset: Optional[Any],
    current_fold_num: int,
    args: Any,
) -> Dict[str, Any]:
    """
    Trains a model for a single fold using PyTorch Lightning.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Optional test dataset.
        current_fold_num: The current fold number.
        args: Namespace object containing command-line arguments and configurations.

    Returns:
        A dictionary containing training and evaluation results for the fold.
    """
    print(
        f"\n{'='*20} Training Fold (PyTorch Lightning): {current_fold_num} | Task: {args.task_type} {'='*20}"
    )

    # 1. Setup Environment and Logger
    fold_results_dir, logger = _setup_environment_and_logger(
        args.results_dir, current_fold_num, args.log_data
    )

    # 2. Initialize DataLoaders
    train_loader, val_loader, test_loader = _initialize_dataloaders(
        train_dataset, val_dataset, test_dataset, args
    )
    train_loader_len = len(train_loader) if train_loader else 0

    # 3. Initialize LightningModule
    lightning_module = _initialize_lightning_module(args)

    # 4. Configure Callbacks
    callbacks, checkpoint_cb = _configure_callbacks(
        args, fold_results_dir, train_loader_len
    )

    # 5. Configure and Initialize PyTorch Lightning Trainer
    trainer = _configure_trainer(args, logger, callbacks, train_loader_len)

    # 6. Start Training
    print("Starting training...")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    print("Training finished.")

    # 7. Final Evaluation on Test Set
    test_metrics, patient_results = _run_final_evaluation(
        trainer, lightning_module, test_loader, checkpoint_cb, fold_results_dir
    )

    # 8. Compile and Return Results
    final_results = _compile_results(
        current_fold_num, args, checkpoint_cb, trainer, test_metrics, patient_results
    )

    print(f"Finished training fold {current_fold_num}. Results: {final_results}")
    return final_results
