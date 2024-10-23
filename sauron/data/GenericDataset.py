import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from sauron.utils.WSIObjects import WSIPatcher


class PatchFileDataset(Dataset):
    def __init__(self, directory_path, transform=None):
        self.transform = transform
        self.patch_paths, self.coordinates = self._load_patch_paths(directory_path)

    def _load_patch_paths(self, directory_path):
        patch_paths = []
        coordinates = []
        for filename in tqdm(os.listdir(directory_path), desc="Loading patches"):
            try:
                name, _ = os.path.splitext(filename)
                x_coord, y_coord = map(int, name.split("_")[:2])
                patch_paths.append(os.path.join(directory_path, filename))
                coordinates.append((x_coord, y_coord))
            except (ValueError, IndexError) as e:
                raise ValueError(f"Invalid filename format: {filename}") from e
        return patch_paths, coordinates

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, index):
        try:
            with Image.open(self.patch_paths[index]) as patch:
                patch_array = np.array(patch)
                coord = self.coordinates[index]
        except IndexError as e:
            raise IndexError(f"Index {index} out of range") from e

        if self.transform:
            patch_array = self.transform(patch_array)

        return patch_array, coord


class WholeSlideImageDataset(Dataset):
    def __init__(self, patcher: WSIPatcher, transform=None):
        self.patcher = patcher
        self.transform = transform
        self.cols, self.rows = self.patcher.get_cols_rows()

    def __len__(self):
        return len(self.patcher)

    def __getitem__(self, index):
        try:
            col = index % self.cols
            row = index // self.cols
            tile, x, y = self.patcher.get_tile(col, row)
        except IndexError as e:
            raise IndexError(f"Index {index} out of range") from e

        if self.transform:
            tile = self.transform(tile)

        return tile, (x, y)
