import os
import pickle
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import openslide
import torch
from huggingface_hub import snapshot_download
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from sauron.data.GenericDataset import WholeSlideImageDataset
from sauron.utils.drawing_utils import visualize_tissue
from sauron.utils.gpd_utils import mask_to_geodataframe
from sauron.utils.WSIObjects import WholeSlideImage, wsi_factory


class TissueSegmenter:
    """
    A class for segmenting tissue in whole slide images using a DeepLabV3 model.
    """

    def __init__(
        self,
        model_name: str = "deeplabv3_resnet50_seg_v4.ckpt",
        batch_size: int = 8,
        download_model: bool = True,
        num_workers: int = 8,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the TissueSegmenter.

        Args:
            model_name (str): The name of the model checkpoint file.
            batch_size (int): Batch size for processing.
            download_model (bool): Whether to download the model if not present.
            num_workers (int): Number of worker threads for data loading.
            output_dir (Optional[str]): Directory to save output files.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.download_model = download_model
        self.num_workers = num_workers
        self.output_dir = output_dir
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        """
        Load the DeepLabV3 model with the specified weights.

        Returns:
            nn.Module: The loaded and initialized model.
        """
        model = torch.hub.load("pytorch/vision:v0.10.0", "deeplabv3_resnet50")
        model.classifier[4] = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1)

        model_dir = Path(__file__).resolve().parents[3] / "models"

        if self.download_model:
            snapshot_download(
                repo_id="MahmoodLab/hest-tissue-seg",
                repo_type="model",
                local_dir=model_dir,
                allow_patterns=self.model_name,
            )

        weights_path = model_dir / self.model_name

        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at {weights_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(weights_path, map_location=device)

        # Clean up state dict
        state_dict = {
            key.replace("model.", ""): value
            for key, value in checkpoint["state_dict"].items()
            if "aux" not in key
        }
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        return model

    def segment_tissue(
        self,
        wsi: Union[np.ndarray, openslide.OpenSlide, WholeSlideImage],
        pixel_size: float,
        save_basename: Optional[str] = None,
        fast_mode: bool = False,
        output_pixel_size: float = 1.0,
        patch_size_micrometers: int = 512,
    ) -> gpd.GeoDataFrame:
        """
        Segment tissue regions from a whole slide image.

        Args:
            wsi (Union[np.ndarray, openslide.OpenSlide, WSI]): The whole slide image to process.
            pixel_size (float): Pixel size of the input image.
            save_basename (Optional[str]): Base name for saving output files.
            fast_mode (bool): If True, enables fast processing at lower resolution.
            output_pixel_size (float): Desired pixel size for output.
            patch_size_micrometers (int): Size of patches in micrometers.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing the segmented tissue contours.
        """
        if fast_mode and output_pixel_size == 1.0:
            output_pixel_size = 2.0

        deeplab_patch_size = 512
        scale_factor = pixel_size / output_pixel_size
        source_patch_size = round(patch_size_micrometers / scale_factor)

        wsi = wsi_factory(wsi)
        patcher = wsi.create_patcher(deeplab_patch_size, pixel_size, output_pixel_size)

        eval_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        dataset = WholeSlideImageDataset(patcher, eval_transforms)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

        cols, rows = patcher.get_cols_rows()
        mask_width = deeplab_patch_size * cols
        mask_height = deeplab_patch_size * rows
        stitched_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
        scaling_ratio = deeplab_patch_size / source_patch_size

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.inference_mode():
            for images, coordinates in tqdm(dataloader, total=len(dataloader)):
                images = images.to(device)
                outputs = self.model(images)["out"]
                predictions = outputs.argmax(dim=1).byte().cpu().numpy()
                x_coords, y_coords = coordinates[0], coordinates[1]

                for pred_mask, x, y in zip(predictions, x_coords, y_coords):
                    x_scaled = round(x.item() * scaling_ratio)
                    y_scaled = round(y.item() * scaling_ratio)
                    y_end = min(y_scaled + deeplab_patch_size, mask_height)
                    x_end = min(x_scaled + deeplab_patch_size, mask_width)
                    mask_slice = pred_mask[: y_end - y_scaled, : x_end - x_scaled]
                    stitched_mask[y_scaled:y_end, x_scaled:x_end] += mask_slice

        binary_mask = (stitched_mask > 0).astype(np.uint8)
        tissue_contours = mask_to_geodataframe(
            binary_mask,
            max_holes=5,
            pixel_size=pixel_size,
            contour_scale=1 / scaling_ratio,
        )

        if self.output_dir and save_basename:
            self._save_outputs(wsi, tissue_contours, save_basename)

        return tissue_contours

    def _save_outputs(
        self,
        wsi: WholeSlideImage,
        tissue_contours: gpd.GeoDataFrame,
        save_basename: str,
    ):
        """
        Save segmentation outputs to files.

        Args:
            wsi (WSI): The whole slide image object.
            tissue_contours (gpd.GeoDataFrame): GeoDataFrame of tissue contours.
            save_basename (str): Base name for output files.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        # Save visualization
        vis_image = visualize_tissue(
            wsi, tissue_contours, line_thickness=5, target_width=1000
        )
        vis_path = os.path.join(self.output_dir, f"{save_basename}_tissue_mask.jpeg")
        vis_image.save(vis_path)

        # Save GeoJSON
        geojson_path = os.path.join(
            self.output_dir, f"{save_basename}_tissue_mask.geojson"
        )
        tissue_contours.to_file(geojson_path, driver="GeoJSON")

        # Save pickle
        pickle_path = os.path.join(self.output_dir, f"{save_basename}_tissue_mask.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(tissue_contours, f)
