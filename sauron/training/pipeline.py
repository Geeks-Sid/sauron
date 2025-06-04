# utils/training_utils.py

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from sauron.data.data_utils import get_dataloader
from sauron.losses.surv_loss import (
    CoxSurvLoss,
    CrossEntropySurvLoss,
    NLLSurvLoss,
)

# from sauron.utils.survival_utils import make_weights_for_balanced_classes_split # Not used in snippet
from sauron.models.models_factory import initialize_mil_model

# --- Utility Imports ---
from sauron.utils.callbacks import EarlyStopping
from sauron.utils.generic_utils import calculate_error
from sauron.utils.optimizers import get_optim

from .trainer import Trainer


# --- Main Fold Training Function (Orchestrator) ---
def train_fold(
    train_dataset: Any,
    val_dataset: Any,
    cur_fold_num: int,
    args: Any,
    experiment_base_results_dir: str,
) -> Union[
    Tuple[
        Optional[Dict[str, Any]], float, float, float, float
    ],  # Classification (non-kfold)
    Tuple[Optional[Dict[str, Any]], float, float],  # Survival (non-kfold)
    float,  # k-fold (val_metric)
]:
    task_type = args.task_type
    print(f"\n{'='*20} Training Fold: {cur_fold_num} | Task: {task_type} {'='*20}")

    results_dir_fold = os.path.join(args.results_dir, str(cur_fold_num))
    os.makedirs(results_dir_fold, exist_ok=True)
    writer = (
        SummaryWriter(log_dir=results_dir_fold, flush_secs=15)
        if args.log_data
        else None
    )

    if args.k_fold:
        train_split, val_split = train_dataset, val_dataset
        test_split = None
        print(f"K-Fold Training: Fold {cur_fold_num}")
    else:
        train_split, val_split, test_split = train_dataset, val_dataset, val_dataset
        splits_file = os.path.join(results_dir_fold, f"splits_{cur_fold_num}.csv")
        # save_splits might need to handle dataset types or have specific versions
        # save_splits(datasets, ["train", "val", "test"], splits_file)
        print(f"Standard Training: Fold {cur_fold_num}")
        if test_split:
            print(f"Test set size: {len(test_split)}")

    print(f"Train set size: {len(train_split)}")
    print(f"Val set size  : {len(val_split)}")

    # Use args.batch_size for training, and typically 1 for MIL validation/testing
    # Allow overriding val/test batch_size via args if necessary, but default to 1
    val_test_batch_size = getattr(args, "val_test_batch_size", 1)

    train_loader = get_dataloader(
        train_split,
        training=True,
        weighted=args.weighted_sample,
        batch_size=args.batch_size,
    )
    val_loader = get_dataloader(
        val_split, training=False, batch_size=val_test_batch_size
    )
    test_loader = (
        get_dataloader(test_split, training=False, batch_size=val_test_batch_size)
        if not args.k_fold and test_split
        else None
    )

    model = initialize_mil_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hasattr(model, "relocate") and callable(
        model.relocate
    ):  # Custom relocate method
        print("Calling model.relocate()...")
        model.relocate()  # This might do more than just .to(device)
    else:
        model.to(device)

    if task_type == "classification":
        loss_fn = nn.CrossEntropyLoss()
    elif task_type == "survival":
        alpha_surv = getattr(args, "alpha_surv", 0.0)  # Default alpha if not in args
        if args.bag_loss == "ce_surv":
            loss_fn = CrossEntropySurvLoss(alpha=alpha_surv)
        elif args.bag_loss == "nll_surv":
            loss_fn = NLLSurvLoss(alpha=alpha_surv)
        elif args.bag_loss == "cox_surv":
            loss_fn = CoxSurvLoss()  # Cox usually doesn't use alpha
        else:
            raise NotImplementedError(f"Survival loss {args.bag_loss} not implemented.")
    else:
        raise ValueError(f"Invalid task_type: {task_type}")

    optimizer = get_optim(model, args)
    trainer = Trainer(
        model, optimizer, loss_fn, device, task_type, args.n_classes, args, writer
    )

    early_stopping_cb = None
    if args.early_stopping:
        best_model_path = os.path.join(
            results_dir_fold, f"s_{cur_fold_num}_best_model.pt"
        )

        es_monitor_metric_name = "metric"  # Default to 'metric' (AUC/C-Index)
        es_mode = "max"
        if task_type == "classification":
            # Allow overriding to monitor loss for classification
            if getattr(args, "early_stop_target", "metric").lower() == "loss":
                es_monitor_metric_name = "loss"
                es_mode = "min"
        # For survival, it's always 'metric' (C-Index) and 'max'

        early_stopping_cb = EarlyStopping(
            patience=getattr(args, "es_patience", 20),
            stop_epoch=getattr(args, "es_stop_epoch", 50),
            verbose=True,
            mode=es_mode,
            stop_metric=es_monitor_metric_name,  # 'loss' or 'metric'
            ckpt_path=best_model_path,
        )

    trainer.fit(
        train_loader,
        val_loader,
        early_stopping_cb=early_stopping_cb,
        max_epochs=args.max_epochs,
    )

    print("\n--- Final Evaluation (using best/last model) ---")
    # Evaluate on validation set with the final (best) model
    val_final_metric, val_final_loss, _ = trainer._evaluate(
        val_loader,
        epoch_log_id=trainer.current_epoch,  # Log against last epoch or a summary step
        eval_survival_loss_alpha=0.0,
        results_prefix="Val_Final",
        collect_patient_results=False,
    )
    if writer:
        writer.add_scalar(f"final_eval/val_loss", val_final_loss, trainer.current_epoch)
        writer.add_scalar(
            f"final_eval/val_{'auc' if task_type=='classification' else 'c_index'}",
            val_final_metric,
            trainer.current_epoch,
        )

    if not args.k_fold and test_loader:
        test_results_dict, test_final_metric, test_final_loss = trainer._evaluate(
            test_loader,
            epoch_log_id=trainer.current_epoch,
            eval_survival_loss_alpha=0.0,
            results_prefix="Test_Final",
            collect_patient_results=True,  # Collect for test
        )
        if writer:
            writer.add_scalar(
                f"final_eval/test_loss", test_final_loss, trainer.current_epoch
            )
            writer.add_scalar(
                f"final_eval/test_{'auc' if task_type=='classification' else 'c_index'}",
                test_final_metric,
                trainer.current_epoch,
            )

        if task_type == "classification":
            test_acc = 0.0
            # Calculate test accuracy from results_dict if available
            if test_results_dict:
                try:
                    preds = torch.tensor(
                        [res["pred"] for res in test_results_dict.values()]
                    )
                    labels = torch.tensor(
                        [res["label"] for res in test_results_dict.values()]
                    )
                    if len(preds) > 0:
                        test_acc = 1.0 - calculate_error(preds, labels)
                        print(f"  Final Test Accuracy: {test_acc:.4f}")
                        if writer:
                            writer.add_scalar(
                                "final_eval/test_acc", test_acc, trainer.current_epoch
                            )
                except KeyError:
                    print(
                        "Warning: 'pred' or 'label' key missing in test_results_dict for accuracy calculation."
                    )

            # Original return: results_dict, test_metric, val_metric, test_acc, val_acc (val_error actually)
            # Val acc/error is not directly returned by _evaluate, would need dedicated calculation.
            # Let's return what we have: test_results, test_metric (AUC), val_metric (AUC), test_acc, placeholder for val_acc
            val_acc_placeholder = 0.0  # Needs to be calculated if required.
            if writer:
                writer.close()
            return (
                test_results_dict,
                test_final_metric,
                val_final_metric,
                test_acc,
                val_acc_placeholder,
            )
        else:  # Survival
            if writer:
                writer.close()
            return test_results_dict, test_final_metric, val_final_metric
    else:  # k-fold (or no test set)
        if writer:
            writer.close()
        return val_final_metric  # Return val_metric for k-fold summary
