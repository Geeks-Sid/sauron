from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import mode
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset


class WSIClassificationDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        shuffle: bool = False,
        random_seed: int = 7,
        verbose: bool = True,
        label_mapping: Optional[Dict[str, int]] = None,
        filter_criteria: Optional[Dict[str, List[str]]] = None,
        ignore_labels: Optional[List[str]] = None,
        patient_stratification: bool = False,
        label_column: Optional[str] = None,
        patient_label_aggregation: str = "max",
    ):
        self.label_mapping = label_mapping or {}
        self.num_classes = len(set(self.label_mapping.values()))
        self.random_seed = random_seed
        self.verbose = verbose
        self.patient_stratification = patient_stratification
        self.train_indices: Optional[List[int]] = None
        self.val_indices: Optional[List[int]] = None
        self.test_indices: Optional[List[int]] = None
        self.data_directory: Optional[str] = None
        self.label_column = label_column or "label"

        # Load and preprocess slide data
        try:
            slide_data = pd.read_csv(csv_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"CSV file not found: {csv_path}") from e

        slide_data = self._filter_data(slide_data, filter_criteria)
        slide_data = self._prepare_data(slide_data, ignore_labels)

        if shuffle:
            slide_data = slide_data.sample(
                frac=1, random_state=random_seed
            ).reset_index(drop=True)

        self.slide_data = slide_data
        self._aggregate_patient_data(patient_label_aggregation)
        self._prepare_class_indices()

        if verbose:
            self._print_summary()

        self.data_cache: Dict[str, torch.Tensor] = {}

    def _prepare_class_indices(self) -> None:
        self.patient_class_indices = [
            np.where(self.patient_data["label"] == cls_label)[0]
            for cls_label in range(self.num_classes)
        ]
        self.slide_class_indices = [
            np.where(self.slide_data["label"] == cls_label)[0]
            for cls_label in range(self.num_classes)
        ]

    def _aggregate_patient_data(self, aggregation_method: str = "max") -> None:
        patients = self.slide_data["case_id"].unique()
        patient_labels = []

        for patient_id in patients:
            labels = self.slide_data.loc[
                self.slide_data["case_id"] == patient_id, "label"
            ].values
            if aggregation_method == "max":
                aggregated_label = labels.max()
            elif aggregation_method == "majority":
                aggregated_label = mode(labels).mode[0]
            else:
                raise ValueError(
                    f"Invalid patient_label_aggregation method: {aggregation_method}"
                )
            patient_labels.append(aggregated_label)

        self.patient_data = pd.DataFrame({"case_id": patients, "label": patient_labels})

    def _prepare_data(
        self,
        data: pd.DataFrame,
        ignore_labels: Optional[List[str]],
    ) -> pd.DataFrame:
        data = data.copy()
        if self.label_column != "label":
            data["label"] = data[self.label_column]

        if ignore_labels:
            data = data[~data["label"].isin(ignore_labels)].reset_index(drop=True)
        data["label"] = data["label"].map(self.label_mapping)

        return data

    def _filter_data(
        self, data: pd.DataFrame, filter_criteria: Optional[Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        if filter_criteria:
            mask = pd.Series(True, index=data.index)
            for column, values in filter_criteria.items():
                mask &= data[column].isin(values)
            data = data[mask].reset_index(drop=True)
        return data

    def __len__(self) -> int:
        return (
            len(self.patient_data)
            if self.patient_stratification
            else len(self.slide_data)
        )

    def _print_summary(self) -> None:
        print(f"Label column: {self.label_column}")
        print(f"Label mapping: {self.label_mapping}")
        print(f"Number of classes: {self.num_classes}")
        print("Slide-level counts:")
        print(self.slide_data["label"].value_counts(sort=False))
        for cls_label in range(self.num_classes):
            print(
                f"Patient-level samples in class {cls_label}: {len(self.patient_class_indices[cls_label])}"
            )
            print(
                f"Slide-level samples in class {cls_label}: {len(self.slide_class_indices[cls_label])}"
            )

    def create_splits(self, num_folds: int = 3) -> None:
        # Create initial test set (10% of patients)
        train_val_patients, test_patients = train_test_split(
            self.patient_data,
            test_size=0.1,
            stratify=self.patient_data["label"],
            random_state=self.random_seed,
        )

        # Create K-fold splits on the remaining patients
        skf = StratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=self.random_seed
        )
        self.splits = list(skf.split(train_val_patients, train_val_patients["label"]))

        # Store patient IDs for splits
        self.patient_splits = {
            "test": test_patients["case_id"].tolist(),
            "train_val": train_val_patients["case_id"].tolist(),
        }

    def set_splits(self, fold_index: int = 0) -> None:
        if not hasattr(self, "patient_splits"):
            raise ValueError(
                "Splits have not been created. Call create_splits() first."
            )

        train_indices, val_indices = self.splits[fold_index]
        train_val_patients = self.patient_data[
            self.patient_data["case_id"].isin(self.patient_splits["train_val"])
        ]

        train_patients = train_val_patients.iloc[train_indices]
        val_patients = train_val_patients.iloc[val_indices]

        # Map patient IDs to slide indices
        self.train_indices = self.slide_data[
            self.slide_data["case_id"].isin(train_patients["case_id"])
        ].index.tolist()
        self.val_indices = self.slide_data[
            self.slide_data["case_id"].isin(val_patients["case_id"])
        ].index.tolist()
        self.test_indices = self.slide_data[
            self.slide_data["case_id"].isin(self.patient_splits["test"])
        ].index.tolist()

    def get_num_splits(self) -> int:
        return len(self.splits)

    def save_splits(self, filename: str) -> None:
        if not all(
            hasattr(self, attr)
            for attr in ["train_indices", "val_indices", "test_indices"]
        ):
            raise ValueError("Splits have not been set. Call set_splits() first.")

        splits = {
            "train": self.slide_data.loc[self.train_indices, "case_id"],
            "val": self.slide_data.loc[self.val_indices, "case_id"],
            "test": self.slide_data.loc[self.test_indices, "case_id"],
        }
        df = pd.DataFrame(splits)
        df.to_csv(filename, index=False)
        print(f"Splits saved to {filename}")

    def get_splits(
        self,
        backbone: str,
        patch_size: str = "",
        use_indices: bool = True,
        splits_csv_path: Optional[str] = None,
    ) -> Tuple[
        Optional["DatasetSplit"], Optional["DatasetSplit"], Optional["DatasetSplit"]
    ]:
        if use_indices:
            if (
                self.train_indices is None
                or self.val_indices is None
                or self.test_indices is None
            ):
                raise ValueError(
                    "Splits have not been created. Call create_splits() and set_splits() first."
                )

            train_split = self._create_dataset_split(
                self.train_indices, backbone, patch_size
            )
            val_split = self._create_dataset_split(
                self.val_indices, backbone, patch_size
            )
            test_split = self._create_dataset_split(
                self.test_indices, backbone, patch_size
            )
        else:
            if not splits_csv_path:
                raise ValueError(
                    "splits_csv_path must be provided when use_indices is False"
                )
            splits_df = pd.read_csv(splits_csv_path)
            train_split = self._get_split_from_df("train", splits_df)
            val_split = self._get_split_from_df("val", splits_df)
            test_split = self._get_split_from_df("test", splits_df)

        return train_split, val_split, test_split

    def _create_dataset_split(
        self, indices: List[int], backbone: str, patch_size: str
    ) -> "DatasetSplit":
        split_data = self.slide_data.loc[indices].reset_index(drop=True)
        split_dataset = DatasetSplit(
            split_data, data_directory=self.data_directory, num_classes=self.num_classes
        )
        split_dataset.set_backbone(backbone)
        split_dataset.set_patch_size(patch_size)
        return split_dataset

    def _get_split_from_df(
        self, split_name: str, splits_df: pd.DataFrame
    ) -> "DatasetSplit":
        if split_name not in ["train", "val", "test"]:
            raise ValueError("split_name must be 'train', 'val', or 'test'")

        split_case_ids = splits_df[split_name].dropna().tolist()
        split_slide_data = self.slide_data[
            self.slide_data["case_id"].isin(split_case_ids)
        ].reset_index(drop=True)

        return DatasetSplit(
            split_slide_data,
            data_directory=self.data_directory,
            num_classes=self.num_classes,
        )

    def summarize_splits(self, return_summary: bool = False) -> Optional[pd.DataFrame]:
        if return_summary:
            class_labels = [
                key
                for key, value in sorted(
                    self.label_mapping.items(), key=lambda item: item[1]
                )
            ]
            summary_df = pd.DataFrame(
                np.zeros((len(class_labels), 3), dtype=int),
                index=class_labels,
                columns=["train", "val", "test"],
            )

        for split_name in ["train", "val", "test"]:
            indices = getattr(self, f"{split_name}_indices")
            if indices is None:
                continue
            print(f"\nNumber of {split_name} samples: {len(indices)}")
            labels = self.slide_data.loc[indices, "label"]
            label_counts = labels.value_counts().sort_index()
            for cls_label, count in label_counts.items():
                print(f"Number of samples in class {cls_label}: {count}")
                if return_summary:
                    class_name = next(
                        key
                        for key, value in self.label_mapping.items()
                        if value == cls_label
                    )
                    summary_df.loc[class_name, split_name] = count

        # Ensure splits are mutually exclusive
        if self.train_indices and self.val_indices and self.test_indices:
            assert not set(self.train_indices) & set(self.val_indices)
            assert not set(self.train_indices) & set(self.test_indices)
            assert not set(self.val_indices) & set(self.test_indices)

        return summary_df if return_summary else None


class WSIMILDataset(WSIClassificationDataset):
    def __init__(self, data_directory: Union[str, Dict[str, str]], **kwargs):
        super().__init__(**kwargs)
        self.data_directory = data_directory
        self.use_hdf5 = False
        self.backbone: Optional[str] = None
        self.patch_size: str = ""
        self.data_cache: Dict[str, torch.Tensor] = {}

    def load_from_hdf5(self, use_hdf5: bool) -> None:
        self.use_hdf5 = use_hdf5

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, np.ndarray]]:
        slide_id = self.slide_data.at[idx, "slide_id"]
        label = self.slide_data.at[idx, "label"]

        if isinstance(self.data_directory, dict):
            source = self.slide_data.at[idx, "source"]
            data_dir = self.data_directory[source]
        elif self.data_directory is None:
            data_dir = self.slide_data.at[idx, "dir"]
        else:
            data_dir = self.data_directory

        if not self.use_hdf5:
            patch_subdir = "" if self.patch_size == "512" else self.patch_size
            file_path = os.path.join(
                data_dir, patch_subdir, "pt_files", self.backbone, f"{slide_id}.pt"
            )

            try:
                features = self.data_cache.get(file_path)
                if features is None:
                    features = torch.load(file_path)
                    if getattr(self, "cache_enabled", False):
                        self.data_cache[file_path] = features
                return features, label
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Feature file not found: {file_path}") from e

        else:
            file_path = os.path.join(data_dir, "h5_files", f"{slide_id}.h5")
            try:
                with h5py.File(file_path, "r") as hdf5_file:
                    features = torch.from_numpy(hdf5_file["features"][:])
                    coordinates = hdf5_file["coords"][:]
                return features, label, coordinates
            except OSError as e:
                raise OSError(f"HDF5 file not found or corrupted: {file_path}") from e

    def set_backbone(self, backbone: str) -> None:
        self.backbone = backbone

    def set_patch_size(self, size: str) -> None:
        self.patch_size = size


class DatasetSplit(WSIMILDataset):
    def __init__(
        self,
        slide_data: pd.DataFrame,
        data_directory: Optional[str] = None,
        num_classes: int = 2,
    ):
        self.use_hdf5 = False
        self.slide_data = slide_data
        self.data_directory = data_directory
        self.num_classes = num_classes
        self.slide_class_indices = [
            np.where(self.slide_data["label"] == cls_label)[0]
            for cls_label in range(self.num_classes)
        ]
        self.data_cache: Dict[str, torch.Tensor] = {}
        self.backbone: Optional[str] = None
        self.patch_size: Optional[str] = None
        self.cache_enabled: bool = False

    def __len__(self) -> int:
        return len(self.slide_data)

    def set_backbone(self, backbone: str) -> None:
        print(f"Setting backbone: {backbone}")
        self.backbone = backbone

    def set_patch_size(self, size: str) -> None:
        print(f"Setting patch size: {size}")
        self.patch_size = size

    def preload_data(self, num_threads: int = 8) -> None:
        self.cache_enabled = True
        indices = list(range(len(self)))
        from multiprocessing.pool import ThreadPool

        with ThreadPool(num_threads) as pool:
            pool.map(self.__getitem__, indices)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, np.ndarray]]:
        return super().__getitem__(idx)
