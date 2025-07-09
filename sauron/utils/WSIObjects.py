# sauron/feature_extraction/wsi/base.py
from __future__ import annotations

import os
import warnings
from typing import List, Literal, Optional, Tuple

import geopandas as gpd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from sauron.feature_extraction.utils.geo_utils import (
    mask_to_gdf,
    overlay_gdf_on_thumbnail,
)
from sauron.feature_extraction.utils.io import (
    get_num_workers,
    read_coords,
    read_coords_legacy,
    save_h5,
)
from sauron.feature_extraction.wsi.dataset import WSIPatcherDataset
from sauron.feature_extraction.wsi.patching import WSIPatcher

ReadMode = Literal["pil", "numpy"]


class WSI:
    """
    The `WSI` class provides an interface to work with Whole Slide Images (WSIs).
    It supports lazy initialization, metadata extraction, tissue segmentation,
    patching, and feature extraction.
    """

    def __init__(
        self,
        slide_path: str,
        name: Optional[str] = None,
        tissue_seg_path: Optional[str] = None,
        custom_mpp_keys: Optional[List[str]] = None,
        lazy_init: bool = True,
        mpp: Optional[float] = None,
        max_workers: Optional[int] = None,
        original_path: Optional[str] = None,
    ):
        self.slide_path = slide_path
        self.original_path = original_path or slide_path
        if name is None:
            self.name, self.ext = os.path.splitext(os.path.basename(slide_path))
        else:
            self.name, self.ext = os.path.splitext(name)
        self.tissue_seg_path = tissue_seg_path
        self.custom_mpp_keys = custom_mpp_keys

        self.width, self.height = None, None
        self.mpp = mpp
        self.mag = None
        self.lazy_init = lazy_init
        self.max_workers = max_workers

        if not self.lazy_init:
            self._lazy_initialize()
        else:
            self.lazy_init = not self.lazy_init

    def __repr__(self) -> str:
        if self.lazy_init:
            return f"<width={self.width}, height={self.height}, backend={self.__class__.__name__}, mpp={self.mpp}, mag={self.mag}>"
        else:
            return f"<name={self.name}>"

    def _lazy_initialize(self) -> None:
        if not self.lazy_init:
            self.img = None
            self.dimensions = None
            self.width, self.height = None, None
            self.level_count = None
            self.level_downsamples = None
            self.level_dimensions = None
            self.properties = None
            self.mag = None
            if self.tissue_seg_path is not None:
                try:
                    self.gdf_contours = gpd.read_file(self.tissue_seg_path)
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"Tissue segmentation file not found: {self.tissue_seg_path}"
                    )

    def create_patcher(
        self,
        patch_size: int,
        src_mag: Optional[int] = None,
        dst_mag: Optional[int] = None,
        overlap: int = 0,
        mask: Optional[gpd.GeoDataFrame] = None,
        coords_only: bool = False,
        custom_coords: Optional[np.ndarray] = None,
        threshold: float = 0.15,
        pil: bool = False,
    ) -> WSIPatcher:
        return WSIPatcher(
            self,
            patch_size,
            src_mag=src_mag,
            dst_mag=dst_mag,
            overlap=overlap,
            mask=mask,
            coords_only=coords_only,
            custom_coords=custom_coords,
            threshold=threshold,
            pil=pil,
        )

    def _fetch_magnification(self, custom_mpp_keys: Optional[List[str]] = None) -> int:
        if self.mpp is None:
            mpp_x = self._fetch_mpp(custom_mpp_keys)
        else:
            mpp_x = self.mpp

        if mpp_x is not None:
            if mpp_x < 0.16:
                return 80
            elif mpp_x < 0.2:
                return 60
            elif mpp_x < 0.3:
                return 40
            elif mpp_x < 0.6:
                return 20
            elif mpp_x < 1.2:
                return 10
            elif mpp_x < 2.4:
                return 5
            else:
                raise ValueError(
                    f"Identified mpp is very low: mpp={mpp_x}. Most WSIs are at 20x, 40x magnfication."
                )

    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def segment_tissue(
        self,
        segmentation_model: torch.nn.Module,
        target_mag: int = 10,
        holes_are_tissue: bool = True,
        job_dir: Optional[str] = None,
        batch_size: int = 16,
        device: str = "cuda:0",
        verbose=False,
    ) -> str:
        self._lazy_initialize()
        segmentation_model.to(device)
        max_dimension = 1000
        if self.width > self.height:
            thumbnail_width = max_dimension
            thumbnail_height = int(thumbnail_width * self.height / self.width)
        else:
            thumbnail_height = max_dimension
            thumbnail_width = int(thumbnail_height * self.width / self.height)
        thumbnail = self.get_thumbnail((thumbnail_width, thumbnail_height))

        destination_mpp = 10 / target_mag
        patcher = self.create_patcher(
            patch_size=segmentation_model.input_size,
            src_mag=self.mag,
            dst_mag=target_mag,
            mask=self.gdf_contours if hasattr(self, "gdf_contours") else None,
        )
        precision = segmentation_model.precision
        eval_transforms = segmentation_model.eval_transforms
        dataset = WSIPatcherDataset(patcher, eval_transforms)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=get_num_workers(batch_size, max_workers=self.max_workers),
            pin_memory=True,
        )

        mpp_reduction_factor = self.mpp / destination_mpp
        width, height = self.get_dimensions()
        width, height = (
            int(round(width * mpp_reduction_factor)),
            int(round(height * mpp_reduction_factor)),
        )
        predicted_mask = np.zeros((height, width), dtype=np.uint8)

        dataloader = tqdm(dataloader) if verbose else dataloader

        for imgs, (xcoords, ycoords) in dataloader:
            imgs = imgs.to(device, dtype=precision)
            with torch.autocast(
                device_type=device.split(":")[0],
                dtype=precision,
                enabled=(precision != torch.float32),
            ):
                preds = segmentation_model(imgs).cpu().numpy()

            x_starts = np.clip(
                np.round(xcoords.numpy() * mpp_reduction_factor).astype(int),
                0,
                width - 1,
            )
            y_starts = np.clip(
                np.round(ycoords.numpy() * mpp_reduction_factor).astype(int),
                0,
                height - 1,
            )
            x_ends = np.clip(x_starts + segmentation_model.input_size, 0, width)
            y_ends = np.clip(y_starts + segmentation_model.input_size, 0, height)

            for i in range(len(preds)):
                x_start, x_end = x_starts[i], x_ends[i]
                y_start, y_end = y_starts[i], y_ends[i]
                if x_start >= x_end or y_start >= y_end:
                    continue
                patch_pred = preds[i][: y_end - y_start, : x_end - x_start]
                predicted_mask[y_start:y_end, x_start:x_end] += patch_pred

        predicted_mask = (predicted_mask > 0).astype(np.uint8) * 255

        thumbnail_saveto = os.path.join(job_dir, "thumbnails", f"{self.name}.jpg")
        os.makedirs(os.path.dirname(thumbnail_saveto), exist_ok=True)
        thumbnail.save(thumbnail_saveto)

        gdf_saveto = os.path.join(job_dir, "contours_geojson", f"{self.name}.geojson")
        os.makedirs(os.path.dirname(gdf_saveto), exist_ok=True)
        gdf_contours = mask_to_gdf(
            mask=predicted_mask,
            max_nb_holes=0 if holes_are_tissue else 20,
            min_contour_area=1000,
            pixel_size=self.mpp,
            contour_scale=1 / mpp_reduction_factor,
        )
        gdf_contours.set_crs("EPSG:3857", inplace=True)
        gdf_contours.to_file(gdf_saveto, driver="GeoJSON")
        self.gdf_contours = gdf_contours
        self.tissue_seg_path = gdf_saveto

        contours_saveto = os.path.join(job_dir, "contours", f"{self.name}.jpg")
        annotated = np.array(thumbnail)
        overlay_gdf_on_thumbnail(
            gdf_contours, annotated, contours_saveto, thumbnail_width / self.width
        )

        return gdf_saveto

    def get_best_level_and_custom_downsample(
        self, downsample: float, tolerance: float = 0.01
    ) -> Tuple[int, float]:
        level_downsamples = self.level_downsamples
        for level, level_downsample in enumerate(level_downsamples):
            if abs(level_downsample - downsample) <= tolerance:
                return level, 1.0

        if downsample >= level_downsamples[0]:
            closest_level = None
            closest_downsample = None
            for level, level_downsample in enumerate(level_downsamples):
                if level_downsample <= downsample:
                    closest_level = level
                    closest_downsample = level_downsample
                else:
                    break
            if closest_level is not None:
                custom_downsample = downsample / closest_downsample
                return closest_level, custom_downsample
        else:
            for level, level_downsample in enumerate(level_downsamples):
                if level_downsample >= downsample:
                    custom_downsample = level_downsample / downsample
                    return level, custom_downsample
        raise ValueError(f"No suitable level found for downsample {downsample}.")

    # Other methods like extract_tissue_coords, extract_patch_features etc. will go here,
    # and they will be very similar to Trident's implementation.
    # ... (omitted for brevity, but you would copy/adapt them from trident.wsi_objects.WSI)

    def extract_tissue_coords(self, **kwargs):
        raise NotImplementedError

    def visualize_coords(self, **kwargs):
        raise NotImplementedError

    def extract_patch_features(self, **kwargs):
        raise NotImplementedError

    def extract_slide_features(self, **kwargs):
        raise NotImplementedError

    def close(self):
        """Release any backend-specific resources."""
        pass
