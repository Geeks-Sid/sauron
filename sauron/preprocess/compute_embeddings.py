import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sauron.data.dataloader import TileDataset
from sauron.utils.hdf5_utils import save_hdf5
from sauron.utils.load_encoders import get_encoder_class


def collate_features(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    coordinates = np.vstack([item[1] for item in batch])
    return images, coordinates


class TileEmbedder:
    def __init__(
        self,
        model_name="conch_ViT-B-16",
        model_repo="hf_hub:MahmoodLab/conch",
        target_patch_size=256,
        target_magnification=20,
        device="cuda",
        precision=None,
        save_directory=None,
        batch_size=64,
        num_workers=8,
    ):
        self.model_name = model_name
        self.model_repo = model_repo
        self.device = device
        self.save_directory = save_directory
        self.target_patch_size = target_patch_size
        self.target_magnification = target_magnification
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.model, self.transform, self.model_precision = self._load_model()
        self.precision = precision if precision is not None else self.model_precision

        self.model.to(self.device)
        self.model.eval()

    def _load_model(self):
        model, transform, model_precision = get_encoder_class(self.model_name)
        return model, transform, model_precision

    def embed_tiles(self, whole_slide_image, tile_contours, slide_name) -> str:
        patch_save_path = os.path.join(
            self.save_directory, "patches", f"{slide_name}_patches.png"
        )
        embedding_save_path = os.path.join(
            self.save_directory, "patch_embeddings", f"{slide_name}.h5"
        )

        dataset = TileDataset(
            wsi=whole_slide_image,
            contours=tile_contours,
            target_patch_size=self.target_patch_size,
            target_magnification=self.target_magnification,
            eval_transform=self.transform,
            save_path=patch_save_path,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_features,
            pin_memory=True,
        )

        for batch_idx, (images, coordinates) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Embedding tiles"
        ):
            images = images.to(self.device, non_blocking=True)
            with torch.inference_mode(), torch.amp.autocast(
                dtype=self.precision, device_type=self.device
            ):
                embeddings = self.model.encode_image(
                    images, proj_contrast=False, normalize=False
                )
            mode = "w" if batch_idx == 0 else "a"
            data_dict = {
                "features": embeddings.cpu().numpy(),
                "coordinates": coordinates,
            }
            save_hdf5(embedding_save_path, data_dict=data_dict, mode=mode)

        return embedding_save_path
