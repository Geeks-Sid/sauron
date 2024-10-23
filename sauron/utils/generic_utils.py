import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter


def seed_everything(seed: int = 7) -> None:
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_splits(
    split_datasets: List[Dataset],
    column_keys: List[str],
    filename: str,
    boolean_style: bool = False,
) -> None:
    splits = [
        split_datasets[i].slide_data["slide_id"] for i in range(len(split_datasets))
    ]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index=True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns=["train", "val", "test"])

    df.to_csv(filename)
    print(f"Splits saved to {filename}")


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def log_results(
    df: pd.DataFrame, args: argparse.Namespace, writer: SummaryWriter
) -> None:
    mean_metrics = df.mean()
    std_metrics = df.std()

    for metric in ["test_auc", "val_auc", "test_acc", "val_acc"]:
        writer.add_scalar(f"mean_{metric}", mean_metrics[metric], args.k_end)
        writer.add_scalar(f"std_{metric}", std_metrics[metric], args.k_end)

    df_append = pd.DataFrame(
        {
            "folds": ["mean", "std"],
            **{
                metric: [mean_metrics[metric], std_metrics[metric]]
                for metric in ["test_auc", "val_auc", "test_acc", "val_acc"]
            },
        }
    )

    final_df = pd.concat([df, df_append])
    save_name = (
        f"summary_partial_{args.k_start}_{args.k_end}.csv"
        if args.k_end - args.k_start != args.k
        else "summary.csv"
    )
    final_df.to_csv(os.path.join(args.results_dir, save_name), index=False)
