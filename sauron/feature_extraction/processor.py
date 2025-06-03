# sauron/feature_extraction/processor.py

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import warnings
from inspect import signature
from typing import Any, Dict, List, Optional, TypeAlias

import geopandas as gpd
import pandas as pd

# External Dependencies
import torch
from tqdm import tqdm

# Sauron Internal Dependencies (New Structure)
from .models.patch_encoders.factory import encoder_factory as patch_encoder_factory
from .models.slide_encoders.factory import encoder_factory as slide_encoder_factory
from .models.slide_encoders.factory import (
    slide_to_patch_encoder_name,  # Mapping needed for slide feature extraction
)
from .utils.config import JSONsaver  # For saving config
from .utils.io import create_lock, is_locked, remove_lock, update_log
from .utils.misc import deprecated  # For deprecated methods
from .wsi.base import WSI  # Import the base WSI class for type hinting
from .wsi.factory import (
    OPENSLIDE_EXTENSIONS,
    PIL_EXTENSIONS,
    WSIReaderType,
    load_wsi,
)

# --- Setup Basic Logging ---
# Configure logging to show informational messages
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Type Aliases for Clarity ---
PathLike: TypeAlias = str | os.PathLike
SegmentationModel: TypeAlias = torch.nn.Module
PatchEncoderModel: TypeAlias = torch.nn.Module
SlideEncoderModel: TypeAlias = torch.nn.Module


class Processor:
    """
    Orchestrates the preprocessing pipeline for Whole Slide Images (WSIs).

    Handles WSI loading, caching, tissue segmentation, patch coordinate extraction,
    and feature extraction at both patch and slide levels using specified models.
    Manages file paths, logging, configuration, and error handling for batch processing.

    Args:
        job_dir (PathLike): Base directory to save all processing results (segmentations,
            patches, features, logs, configs). Must exist.
        wsi_source (PathLike): Directory containing the raw WSI files.
        wsi_ext (Optional[List[str]]): List of file extensions to consider as WSIs
            (e.g., ['.svs', '.ndpi']). If None, uses a default list covering common
            OpenSlide and PIL formats. Defaults to None.
        wsi_cache (Optional[PathLike]): Optional local directory to cache WSIs before
            processing. Useful if `wsi_source` is on a slow network drive.
            Defaults to None.
        clear_cache (bool): If True and `wsi_cache` is provided, delete WSIs from
            the cache after they are processed. Defaults to False.
        skip_errors (bool): If True, log errors encountered during processing a
            single slide and continue with the next. If False, raise the exception
            and stop processing. Defaults to False.
        custom_mpp_keys (Optional[List[str]]): List of custom metadata keys to check
            within WSI properties for retrieving microns-per-pixel (MPP) values.
            Defaults to None.
        custom_list_of_wsis (Optional[PathLike]): Path to a CSV file defining a
            specific list of WSIs to process. The CSV must contain a column named 'wsi'
            (filenames with extensions). It can optionally contain an 'mpp' column to
            override MPP detection. If provided, only WSIs listed in this file and
            found in `wsi_source` will be processed. Defaults to None.
        max_workers (Optional[int]): Maximum number of worker processes for PyTorch
            DataLoaders during feature extraction. If None, a default based on
            CPU cores and batch size is used. Defaults to None.
        reader_type (Optional[WSIReaderType]): Force the use of a specific WSI reading
            backend ('openslide', 'image', 'cucim'). If None, the backend is chosen
            automatically based on the file extension. Defaults to None.

    Raises:
        EnvironmentError: If Python version is less than 3.9.
        AssertionError: If `wsi_ext` is provided but is not a list or if extensions
            do not start with a period.
        ValueError: If `custom_list_of_wsis` CSV does not contain a 'wsi' column,
                    or if other configuration issues arise.
        FileNotFoundError: If specified WSI files, `job_dir`, `wsi_source`, or
                           necessary resources (like the CSV) are missing.
    """

    def __init__(
        self,
        job_dir: PathLike,
        wsi_source: PathLike,
        wsi_ext: Optional[List[str]] = None,
        wsi_cache: Optional[PathLike] = None,
        clear_cache: bool = False,
        skip_errors: bool = False,
        custom_mpp_keys: Optional[List[str]] = None,
        custom_list_of_wsis: Optional[PathLike] = None,
        max_workers: Optional[int] = None,
        reader_type: Optional[WSIReaderType] = None,
    ) -> None:
        # --- Basic Environment and Path Checks ---
        if not (sys.version_info.major >= 3 and sys.version_info.minor >= 9):
            raise EnvironmentError(
                "Sauron Feature Extraction requires Python 3.9 or above. Python 3.10 is recommended."
            )

        if not os.path.isdir(job_dir):
            raise FileNotFoundError(f"Job directory does not exist: {job_dir}")
        if not os.path.isdir(wsi_source):
            raise FileNotFoundError(
                f"WSI source directory does not exist: {wsi_source}"
            )
        if wsi_cache and not os.path.isdir(wsi_cache):
            logger.info(f"Cache directory {wsi_cache} not found. Creating it.")
            os.makedirs(wsi_cache, exist_ok=True)
        if custom_list_of_wsis and not os.path.isfile(custom_list_of_wsis):
            raise FileNotFoundError(
                f"Custom WSI list file not found: {custom_list_of_wsis}"
            )

        # --- Assign Core Attributes ---
        self.job_dir = os.path.abspath(job_dir)
        self.wsi_source = os.path.abspath(wsi_source)
        self.wsi_cache = os.path.abspath(wsi_cache) if wsi_cache else None
        self.wsi_ext = wsi_ext or (list(PIL_EXTENSIONS) + list(OPENSLIDE_EXTENSIONS))
        self.clear_cache = clear_cache
        self.skip_errors = skip_errors
        self.custom_mpp_keys = custom_mpp_keys
        self.max_workers = max_workers
        self.reader_type = reader_type

        # --- Validate WSI Extensions ---
        self._validate_wsi_extensions()

        # --- Identify and Filter Target WSIs ---
        self.wsis: List[WSI] = self._load_and_filter_wsi_objects(custom_list_of_wsis)

        logger.info(f"Processor initialized. Targeting {len(self.wsis)} slides.")
        if self.wsi_cache:
            logger.info(
                f"Using local cache at {self.wsi_cache}. Current files: {len(os.listdir(self.wsi_cache))}"
            )

    def _validate_wsi_extensions(self):
        """Validates the format of the wsi_ext list."""
        if not isinstance(self.wsi_ext, list):
            raise AssertionError(
                f"`wsi_ext` must be a list of file extensions, got {self.wsi_ext} of type {type(self.wsi_ext)}"
            )
        for ext in self.wsi_ext:
            if not ext.startswith("."):
                raise AssertionError(
                    f"Each extension in `wsi_ext` must start with a period (.). Found: {ext}"
                )
            # Store extensions in lowercase for case-insensitive matching
            self.wsi_ext = [e.lower() for e in self.wsi_ext]

    def _load_and_filter_wsi_objects(
        self, custom_list_path: Optional[PathLike]
    ) -> List[WSI]:
        """Identifies WSI files, filters based on CSV if provided, and creates WSI objects."""
        available_slides_in_source = {
            name: os.path.join(self.wsi_source, name)
            for name in os.listdir(self.wsi_source)
            if os.path.splitext(name)[1].lower() in self.wsi_ext
        }
        logger.info(
            f"Found {len(available_slides_in_source)} slides matching extensions in {self.wsi_source}."
        )

        target_slides_info = {}  # Dict[filename, {'path': path, 'mpp': mpp_or_none}]

        if custom_list_path is not None:
            try:
                wsi_df = pd.read_csv(custom_list_path)
            except Exception as e:
                raise ValueError(
                    f"Error reading custom WSI list CSV {custom_list_path}: {e}"
                )

            if "wsi" not in wsi_df.columns:
                raise ValueError(
                    "Custom WSI list CSV must contain a column named 'wsi'."
                )

            has_mpp_column = "mpp" in wsi_df.columns
            processed_filenames = set()

            for _, row in wsi_df.iterrows():
                filename = str(row["wsi"]).strip()
                if not filename:
                    continue

                if filename in processed_filenames:
                    logger.warning(
                        f"Duplicate filename '{filename}' found in CSV. Skipping subsequent entries."
                    )
                    continue
                processed_filenames.add(filename)

                if filename in available_slides_in_source:
                    mpp_value = None
                    if has_mpp_column and pd.notna(row["mpp"]):
                        try:
                            mpp_value = float(row["mpp"])
                        except ValueError:
                            logger.warning(
                                f"Could not parse MPP value '{row['mpp']}' for WSI '{filename}'. Will attempt auto-detection."
                            )
                    target_slides_info[filename] = {
                        "path": available_slides_in_source[filename],
                        "mpp": mpp_value,
                    }
                else:
                    logger.warning(
                        f"WSI '{filename}' listed in CSV but not found in source directory: {self.wsi_source}. Skipping."
                    )
        else:
            # Process all found slides if no custom list
            for filename, filepath in available_slides_in_source.items():
                target_slides_info[filename] = {"path": filepath, "mpp": None}

        # --- Create WSI Objects ---
        wsi_objects = []
        init_log_path = os.path.join(self.job_dir, "_processor_init_log.txt")
        sorted_target_filenames = sorted(target_slides_info.keys())

        for filename in sorted_target_filenames:
            info = target_slides_info[filename]
            wsi_load_path = (
                os.path.join(self.wsi_cache, filename)
                if self.wsi_cache
                else info["path"]
            )

            slide_name_no_ext = os.path.splitext(filename)[0]
            # Standardized path for segmentation results
            tissue_seg_path = os.path.join(
                self.job_dir,
                "segmentation_results",
                "contours_geojson",
                f"{slide_name_no_ext}.geojson",
            )
            if not os.path.exists(tissue_seg_path):
                tissue_seg_path = None

            try:
                slide = load_wsi(
                    slide_path=wsi_load_path,
                    original_path=info["path"],
                    name=filename,
                    tissue_seg_path=tissue_seg_path,
                    custom_mpp_keys=self.custom_mpp_keys,
                    mpp=info["mpp"],
                    max_workers=self.max_workers,
                    reader_type=self.reader_type,
                    lazy_init=True,
                )
                wsi_objects.append(slide)
                update_log(init_log_path, filename, "INFO - WSI object created")
            except Exception as e:
                message = (
                    f"ERROR creating WSI object for {filename} at {wsi_load_path}: {e}"
                )
                update_log(init_log_path, filename, message)
                if self.skip_errors:
                    logger.error(message)
                else:
                    raise RuntimeError(message) from e

        return wsi_objects

    def _get_job_paths(self, job_name: str, sub_dirs: List[str]) -> Dict[str, str]:
        """Helper to create and return paths for a specific processing job."""
        base_dir = os.path.join(self.job_dir, job_name)
        paths = {"base": base_dir}
        for sub in sub_dirs:
            path = os.path.join(base_dir, sub)
            os.makedirs(path, exist_ok=True)
            paths[sub] = path
        paths["config"] = os.path.join(base_dir, f"_config_{job_name}.json")
        paths["log"] = os.path.join(base_dir, f"_log_{job_name}.txt")
        return paths

    def populate_cache(self) -> None:
        """Copies WSI files from the source directory to the local cache directory."""
        if not self.wsi_cache:
            logger.info("No cache directory specified. Skipping cache population.")
            return

        cache_log_path = os.path.join(self.wsi_cache, "_cache_log.txt")
        logger.info(f"Populating cache directory: {self.wsi_cache}")
        progress_bar = tqdm(
            self.wsis, desc="Populating cache", total=len(self.wsis), unit="slide"
        )

        for wsi in progress_bar:
            slide_fullname = wsi.name + wsi.ext
            cache_file_path = os.path.join(self.wsi_cache, slide_fullname)
            source_file_path = wsi.original_path

            progress_bar.set_postfix_str(f"{slide_fullname}")

            if os.path.exists(cache_file_path) and not is_locked(cache_file_path):
                update_log(cache_log_path, slide_fullname, "INFO - Already in cache")
                continue

            if is_locked(cache_file_path):
                update_log(
                    cache_log_path, slide_fullname, "SKIP - Locked by another process"
                )
                continue

            try:
                create_lock(cache_file_path)
                update_log(cache_log_path, slide_fullname, "LOCK - Copying")
                shutil.copy2(source_file_path, cache_file_path)
                update_log(cache_log_path, slide_fullname, "OK - Copied")
            except Exception as e:
                error_msg = f"ERROR copying: {e}"
                update_log(cache_log_path, slide_fullname, error_msg)
                logger.error(f"Failed to copy {source_file_path} to cache: {e}")
                # Attempt cleanup, ignore errors
                try:
                    if os.path.exists(cache_file_path + ".lock"):
                        remove_lock(cache_file_path)
                    if os.path.exists(cache_file_path):  # Remove partially copied file
                        os.remove(cache_file_path)
                except OSError:
                    pass
            finally:
                if os.path.exists(cache_file_path + ".lock"):
                    try:
                        remove_lock(cache_file_path)
                    except OSError as lock_err:
                        logger.warning(
                            f"Could not remove lock file for {cache_file_path} after operation: {lock_err}"
                        )

    def run_segmentation_job(
        self,
        segmentation_model: SegmentationModel,
        seg_mag: int = 10,
        holes_are_tissue: bool = False,
        batch_size: int = 16,
        artifact_remover_model: Optional[SegmentationModel] = None,
        device: str = "cuda:0",
    ) -> str:
        """
        Performs tissue segmentation on the targeted WSIs.

        Uses the provided `segmentation_model` to identify tissue regions.
        Optionally uses an `artifact_remover_model` for refinement. Saves results
        (thumbnails, contours, GeoJSON) in subdirectories under `job_dir`.

        Args:
            segmentation_model: A pre-trained PyTorch model for tissue segmentation.
            seg_mag: Target magnification (e.g., 10 for 10x) for segmentation. Defaults to 10.
            holes_are_tissue: If True, holes within tissue contours are considered tissue.
                              If False, they are excluded. Defaults to False.
            batch_size: Batch size for model inference during segmentation. Defaults to 16.
            artifact_remover_model: Optional second model to refine segmentation, often
                                    used to remove artifacts like pen marks. Defaults to None.
            device: The device for PyTorch computations (e.g., 'cuda:0', 'cpu'). Defaults to 'cuda:0'.

        Returns:
            Absolute path to the directory where GeoJSON contour files are saved.

        Raises:
            RuntimeError: If an error occurs during segmentation and `skip_errors` is False.
        """
        segmentation_dir = os.path.join(self.job_dir, "segmentation_results")
        geojson_dir = os.path.join(segmentation_dir, "contours_geojson")
        contour_img_dir = os.path.join(segmentation_dir, "contours")  # Visualizations
        thumbnail_dir = os.path.join(segmentation_dir, "thumbnails")  # Raw thumbnails
        os.makedirs(geojson_dir, exist_ok=True)
        os.makedirs(contour_img_dir, exist_ok=True)
        os.makedirs(thumbnail_dir, exist_ok=True)

        job_name = "segmentation_results"
        paths = self._get_job_paths(
            job_name, ["contours_geojson", "contours", "thumbnails"]
        )
        geojson_dir = paths["contours_geojson"]
        log_fp = paths["log"]

        # --- Save Configuration ---
        sig = signature(self.run_segmentation_job)
        local_attrs = {k: v for k, v in locals().items() if k in sig.parameters}
        # Add model names if available
        if hasattr(
            segmentation_model, "model_name"
        ):  # Assuming models might have this attr
            local_attrs["segmentation_model_name"] = segmentation_model.model_name
        if artifact_remover_model and hasattr(artifact_remover_model, "model_name"):
            local_attrs["artifact_remover_model_name"] = (
                artifact_remover_model.model_name
            )
        self.save_config(
            saveto=paths["config"],
            local_attrs=local_attrs,
            ignore=["self", "segmentation_model", "artifact_remover_model"],
        )
        logger.info(f"Starting segmentation job. Results will be in {paths['base']}")

        progress_bar = tqdm(
            self.wsis, desc="Segmenting tissue", total=len(self.wsis), unit="slide"
        )

        for wsi in progress_bar:
            slide_fullname = wsi.name + wsi.ext
            geojson_path = os.path.join(geojson_dir, f"{wsi.name}.geojson")
            progress_bar.set_postfix_str(f"{slide_fullname}")

            # --- Pre-computation Checks ---
            if os.path.exists(geojson_path) and not is_locked(geojson_path):
                update_log(log_fp, slide_fullname, "DONE - Already segmented")
                self.cleanup(slide_fullname)
                continue
            if is_locked(geojson_path):
                update_log(log_fp, slide_fullname, "SKIP - Locked")
                continue
            wsi_current_path = (
                os.path.join(self.wsi_cache, slide_fullname)
                if self.wsi_cache
                else wsi.original_path
            )
            if not os.path.exists(
                wsi_current_path
            ):  # Check existence, ignore lock here
                update_log(
                    log_fp,
                    slide_fullname,
                    f"SKIP - WSI not found at {wsi_current_path}",
                )
                continue

            # --- Perform Segmentation ---
            try:
                create_lock(geojson_path)
                update_log(log_fp, slide_fullname, "LOCK - Segmenting")
                wsi._lazy_initialize()  # Ensure WSI is loaded

                generated_geojson_path = wsi.segment_tissue(
                    segmentation_model=segmentation_model,
                    target_mag=seg_mag,
                    holes_are_tissue=holes_are_tissue,
                    job_dir=paths["base"],  # Pass base segmentation dir
                    batch_size=batch_size,
                    device=device,
                    verbose=False,
                )

                if artifact_remover_model is not None:
                    logger.info(f"Applying artifact remover to {slide_fullname}")
                    generated_geojson_path = wsi.segment_tissue(
                        segmentation_model=artifact_remover_model,
                        target_mag=getattr(
                            artifact_remover_model, "target_mag", seg_mag
                        ),
                        holes_are_tissue=False,
                        job_dir=paths["base"],
                        batch_size=batch_size,
                        device=device,
                        verbose=False,
                    )

                # Verify output
                if not os.path.exists(generated_geojson_path):
                    raise FileNotFoundError(
                        f"Segmentation output {generated_geojson_path} not created."
                    )
                try:
                    gdf = gpd.read_file(generated_geojson_path, rows=1)
                    status = "DONE - Segmented"
                    if gdf.empty:
                        status = "WARN - Empty GeoDataFrame"
                        logger.warning(
                            f"Empty segmentation result for {slide_fullname}"
                        )
                    update_log(log_fp, slide_fullname, status)
                except Exception as gdf_err:
                    update_log(
                        log_fp, slide_fullname, f"ERROR reading GeoJSON: {gdf_err}"
                    )
                    raise ValueError(
                        f"Could not read generated GeoJSON: {gdf_err}"
                    ) from gdf_err

            except Exception as e:
                error_msg = f"ERROR during segmentation: {e}"
                update_log(log_fp, slide_fullname, error_msg)
                logger.error(f"Error segmenting {slide_fullname}: {e}")
                if isinstance(e, KeyboardInterrupt):
                    print("Segmentation interrupted.")
                    raise e
                if not self.skip_errors:
                    raise RuntimeError(f"Error segmenting {slide_fullname}: {e}") from e
                # Continue loop if skipping errors
            finally:
                if os.path.exists(geojson_path + ".lock"):
                    try:
                        remove_lock(geojson_path)
                    except OSError as lock_err:
                        logger.warning(
                            f"Could not remove lock {geojson_path}.lock: {lock_err}"
                        )
                self.cleanup(slide_fullname)
                wsi.close()  # Close WSI handle

        logger.info(f"Segmentation job finished. GeoJSONs in: {geojson_dir}")
        return geojson_dir

    def run_patching_job(
        self,
        target_magnification: int,
        patch_size: int,
        overlap: int = 0,
        patch_dir_name: Optional[str] = None,
        visualize: bool = True,
        min_tissue_proportion: float = 0.0,
    ) -> str:
        """Extracts patch coordinates from segmented tissue regions for each WSI."""
        if patch_dir_name is None:
            patch_dir_name = (
                f"patches_{target_magnification}x_{patch_size}px_{overlap}ovlp"
            )

        paths = self._get_job_paths(
            patch_dir_name, ["patches", "visualization"] if visualize else ["patches"]
        )
        coords_h5_dir = paths["patches"]  # HDF5 files go here
        viz_dir = paths.get("visualization")  # Will be None if visualize=False
        log_fp = paths["log"]

        # --- Save Configuration ---
        sig = signature(self.run_patching_job)
        local_attrs = {k: v for k, v in locals().items() if k in sig.parameters}
        self.save_config(
            saveto=paths["config"], local_attrs=local_attrs, ignore=["self"]
        )
        logger.info(f"Starting patching job. Results will be in {paths['base']}")

        progress_bar = tqdm(
            self.wsis,
            desc=f"Extracting patch coordinates ({patch_dir_name})",
            total=len(self.wsis),
            unit="slide",
        )

        for wsi in progress_bar:
            slide_fullname = wsi.name + wsi.ext
            coords_h5_path = os.path.join(coords_h5_dir, f"{wsi.name}_patches.h5")
            progress_bar.set_postfix_str(f"{slide_fullname}")

            # --- Pre-computation Checks ---
            if os.path.exists(coords_h5_path) and not is_locked(coords_h5_path):
                update_log(log_fp, slide_fullname, "DONE - Coords already generated")
                self.cleanup(slide_fullname)
                continue
            if is_locked(coords_h5_path):
                update_log(log_fp, slide_fullname, "SKIP - Locked")
                continue
            wsi_current_path = (
                os.path.join(self.wsi_cache, slide_fullname)
                if self.wsi_cache
                else wsi.original_path
            )
            if not os.path.exists(wsi_current_path):
                update_log(
                    log_fp,
                    slide_fullname,
                    f"SKIP - WSI not found at {wsi_current_path}",
                )
                continue
            segmentation_path = wsi.tissue_seg_path  # Should be set if segmentation ran
            if segmentation_path is None or not os.path.exists(segmentation_path):
                update_log(
                    log_fp, slide_fullname, "SKIP - Segmentation GeoJSON not found"
                )
                continue
            try:  # Check if GeoJSON is empty
                gdf = gpd.read_file(segmentation_path, rows=1)
                if gdf.empty:
                    update_log(
                        log_fp,
                        slide_fullname,
                        "SKIP - Empty GeoDataFrame for segmentation",
                    )
                    continue
            except Exception as gdf_err:
                update_log(log_fp, slide_fullname, f"ERROR reading GeoJSON: {gdf_err}")
                if not self.skip_errors:
                    raise RuntimeError(
                        f"Error reading GeoJSON {segmentation_path}"
                    ) from gdf_err
                continue

            # --- Perform Patching ---
            try:
                create_lock(coords_h5_path)
                update_log(log_fp, slide_fullname, "LOCK - Generating coords")
                wsi._lazy_initialize()  # Ensure WSI loaded

                generated_coords_path = wsi.extract_tissue_coords(
                    target_mag=target_magnification,
                    patch_size=patch_size,
                    save_coords=paths["base"],  # Pass base dir for patching job
                    overlap=overlap,
                    min_tissue_proportion=min_tissue_proportion,
                )

                if not os.path.exists(generated_coords_path):
                    raise FileNotFoundError(
                        f"Coordinate file {generated_coords_path} not created."
                    )

                if viz_dir:
                    wsi.visualize_coords(
                        coords_path=generated_coords_path,
                        save_patch_viz=viz_dir,
                    )
                update_log(log_fp, slide_fullname, "DONE - Coords generated")

            except Exception as e:
                error_msg = f"ERROR during patching: {e}"
                update_log(log_fp, slide_fullname, error_msg)
                logger.error(f"Error patching {slide_fullname}: {e}")
                if isinstance(e, KeyboardInterrupt):
                    print("Patching interrupted.")
                    raise e
                if not self.skip_errors:
                    raise RuntimeError(f"Error patching {slide_fullname}: {e}") from e
            finally:
                if os.path.exists(coords_h5_path + ".lock"):
                    try:
                        remove_lock(coords_h5_path)
                    except OSError as lock_err:
                        logger.warning(
                            f"Could not remove lock {coords_h5_path}.lock: {lock_err}"
                        )
                self.cleanup(slide_fullname)
                wsi.close()

        logger.info(f"Patching job finished. Coordinates in: {coords_h5_dir}")
        return coords_h5_dir  # Return path to HDF5 coordinate files

    @deprecated
    def run_feature_extraction_job(self, *args, **kwargs):
        """Deprecated alias for run_patch_feature_extraction_job."""
        warnings.warn(
            "`run_feature_extraction_job` is deprecated. Use `run_patch_feature_extraction_job` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Map old 'saveto' kwarg to new 'features_dir_name' if present
        if "saveto" in kwargs:
            kwargs["features_dir_name"] = kwargs.pop("saveto")
        return self.run_patch_feature_extraction_job(*args, **kwargs)

    def run_patch_feature_extraction_job(
        self,
        coords_h5_dir: str,  # Dir containing HDF5 patch coord files
        patch_encoder: PatchEncoderModel,
        device: str = "cuda:0",
        saveas: str = "h5",
        batch_limit: int = 512,
        features_dir_name: Optional[str] = None,
    ) -> str:
        """Extracts patch-level features using a specified patch encoder model."""
        # --- Determine Paths ---
        if not os.path.isdir(coords_h5_dir):
            raise FileNotFoundError(f"Coordinates directory not found: {coords_h5_dir}")
        # Assume coords_h5_dir is like .../job_dir/patch_job_name/patches/
        patching_base_dir = os.path.dirname(coords_h5_dir)

        enc_name = getattr(patch_encoder, "enc_name", "custom_encoder")
        if features_dir_name is None:
            features_dir_name = f"features_{enc_name}"

        # Feature files will live alongside the 'patches' dir
        features_base_dir = os.path.join(patching_base_dir, features_dir_name)
        os.makedirs(features_base_dir, exist_ok=True)

        paths = {
            "base": features_base_dir,
            "config": os.path.join(features_base_dir, "_config_patch_features.json"),
            "log": os.path.join(features_base_dir, "_log_patch_features.txt"),
        }
        log_fp = paths["log"]

        # --- Save Configuration ---
        sig = signature(self.run_patch_feature_extraction_job)
        local_attrs = {k: v for k, v in locals().items() if k in sig.parameters}
        local_attrs["patch_encoder_name"] = enc_name
        self.save_config(
            saveto=paths["config"],
            local_attrs=local_attrs,
            ignore=["self", "patch_encoder"],
        )
        logger.info(
            f"Starting patch feature extraction ({features_dir_name}). Results in {paths['base']}"
        )

        progress_bar = tqdm(
            self.wsis,
            desc=f"Extracting patch features ({features_dir_name})",
            total=len(self.wsis),
            unit="slide",
        )

        for wsi in progress_bar:
            slide_fullname = wsi.name + wsi.ext
            coord_h5_path = os.path.join(coords_h5_dir, f"{wsi.name}_patches.h5")
            feature_file_path = os.path.join(features_base_dir, f"{wsi.name}.{saveas}")
            progress_bar.set_postfix_str(f"{slide_fullname}")

            # --- Pre-computation Checks ---
            if os.path.exists(feature_file_path) and not is_locked(feature_file_path):
                update_log(log_fp, slide_fullname, "DONE - Features already extracted")
                self.cleanup(slide_fullname)
                continue
            if is_locked(feature_file_path):
                update_log(log_fp, slide_fullname, "SKIP - Locked")
                continue
            wsi_current_path = (
                os.path.join(self.wsi_cache, slide_fullname)
                if self.wsi_cache
                else wsi.original_path
            )
            if not os.path.exists(wsi_current_path):
                update_log(
                    log_fp,
                    slide_fullname,
                    f"SKIP - WSI not found at {wsi_current_path}",
                )
                continue
            if not os.path.exists(coord_h5_path):
                update_log(
                    log_fp,
                    slide_fullname,
                    f"SKIP - Coordinate file not found: {coord_h5_path}",
                )
                continue

            # --- Perform Feature Extraction ---
            try:
                create_lock(feature_file_path)
                update_log(log_fp, slide_fullname, "LOCK - Extracting patch features")
                wsi._lazy_initialize()  # Ensure WSI loaded

                generated_feature_path = wsi.extract_patch_features(
                    patch_encoder=patch_encoder,
                    coords_path=coord_h5_path,
                    save_features=features_base_dir,  # Pass the target directory
                    device=device,
                    saveas=saveas,
                    batch_limit=batch_limit,
                )

                if not os.path.exists(generated_feature_path):
                    raise FileNotFoundError(
                        f"Feature file {generated_feature_path} not created."
                    )
                update_log(log_fp, slide_fullname, "DONE - Features extracted")

            except Exception as e:
                error_msg = f"ERROR during patch feature extraction: {e}"
                update_log(log_fp, slide_fullname, error_msg)
                logger.error(
                    f"Error extracting patch features for {slide_fullname}: {e}"
                )
                if isinstance(e, KeyboardInterrupt):
                    print("Patch feature extraction interrupted.")
                    raise e
                if not self.skip_errors:
                    raise RuntimeError(
                        f"Error extracting patch features for {slide_fullname}: {e}"
                    ) from e
            finally:
                if os.path.exists(feature_file_path + ".lock"):
                    try:
                        remove_lock(feature_file_path)
                    except OSError as lock_err:
                        logger.warning(
                            f"Could not remove lock {feature_file_path}.lock: {lock_err}"
                        )
                self.cleanup(slide_fullname)
                wsi.close()

        logger.info(
            f"Patch feature extraction finished. Features in: {features_base_dir}"
        )
        return features_base_dir

    def run_slide_feature_extraction_job(
        self,
        slide_encoder: SlideEncoderModel,
        patch_features_dir: str,
        device: str = "cuda:0",
        saveas: str = "h5",
        batch_limit_for_patch_features: int = 512,
        slide_features_dir_name: Optional[str] = None,
    ) -> str:
        """Extracts slide-level features using a specified slide encoder model."""
        # --- Determine Paths and Required Patch Encoder ---
        if not os.path.isdir(patch_features_dir):
            raise FileNotFoundError(
                f"Patch features directory not found: {patch_features_dir}"
            )
        patching_base_dir = os.path.dirname(
            patch_features_dir
        )  # e.g., job_dir/patches.../

        slide_enc_name = getattr(slide_encoder, "enc_name", "custom_slide_encoder")
        required_patch_enc_name = None
        if slide_enc_name.startswith("mean-"):
            required_patch_enc_name = slide_enc_name.split("mean-", 1)[1]
        elif slide_enc_name in slide_to_patch_encoder_name:
            required_patch_enc_name = slide_to_patch_encoder_name[slide_enc_name]

        # Verify input patch_features_dir name consistency (optional check)
        if required_patch_enc_name and not os.path.basename(
            patch_features_dir
        ).endswith(f"features_{required_patch_enc_name}"):
            logger.warning(
                f"Input `patch_features_dir` ('{os.path.basename(patch_features_dir)}') "
                f"might not match expected features for slide encoder '{slide_enc_name}' "
                f"(expected suffix: 'features_{required_patch_enc_name}'). Ensure features are correct."
            )

        # Determine output directory for slide features
        if slide_features_dir_name is None:
            slide_features_dir_name = f"slide_features_{slide_enc_name}"
        slide_features_base_dir = os.path.join(
            patching_base_dir, slide_features_dir_name
        )
        os.makedirs(slide_features_base_dir, exist_ok=True)

        paths = {
            "base": slide_features_base_dir,
            "config": os.path.join(
                slide_features_base_dir, "_config_slide_features.json"
            ),
            "log": os.path.join(slide_features_base_dir, "_log_slide_features.txt"),
        }
        log_fp = paths["log"]

        # --- Auto-generate Patch Features if Missing ---
        if required_patch_enc_name:
            missing_patch_features = any(
                not os.path.exists(os.path.join(patch_features_dir, f"{wsi.name}.h5"))
                for wsi in self.wsis
            )
            if missing_patch_features:
                logger.warning(
                    f"Required patch features ('{required_patch_enc_name}') missing in '{patch_features_dir}'. Attempting generation."
                )
                try:
                    patch_encoder = patch_encoder_factory(required_patch_enc_name)
                    coords_h5_dir = os.path.join(
                        patching_base_dir, "patches"
                    )  # Assumes standard structure
                    if not os.path.isdir(coords_h5_dir):
                        raise FileNotFoundError(
                            f"Coordinate directory '{coords_h5_dir}' needed for auto-generation not found."
                        )

                    # Run the patch feature job, ensuring it saves to the correct input dir for *this* job
                    self.run_patch_feature_extraction_job(
                        coords_h5_dir=coords_h5_dir,
                        patch_encoder=patch_encoder,
                        device=device,
                        saveas="h5",  # Must be h5
                        batch_limit=batch_limit_for_patch_features,
                        features_dir_name=os.path.basename(
                            patch_features_dir
                        ),  # Save to the dir we need
                    )
                    logger.info("Attempted patch feature generation.")
                except Exception as patch_gen_e:
                    raise RuntimeError(
                        f"Failed to auto-generate required patch features ('{required_patch_enc_name}'). "
                        f"Generate manually or ensure they exist in '{patch_features_dir}'. Error: {patch_gen_e}"
                    ) from patch_gen_e
        elif not slide_enc_name.startswith("mean-"):
            logger.warning(
                f"Cannot auto-generate patch features for '{slide_enc_name}' (unknown requirement). Ensure they exist in '{patch_features_dir}'."
            )

        # --- Save Configuration ---
        sig = signature(self.run_slide_feature_extraction_job)
        local_attrs = {k: v for k, v in locals().items() if k in sig.parameters}
        local_attrs["slide_encoder_name"] = slide_enc_name
        self.save_config(
            saveto=paths["config"],
            local_attrs=local_attrs,
            ignore=["self", "slide_encoder"],
        )
        logger.info(
            f"Starting slide feature extraction ({slide_features_dir_name}). Results in {paths['base']}"
        )

        progress_bar = tqdm(
            self.wsis,
            desc=f"Extracting slide features ({slide_features_dir_name})",
            total=len(self.wsis),
            unit="slide",
        )

        for wsi in progress_bar:
            slide_fullname = wsi.name + wsi.ext
            patch_feature_h5_path = os.path.join(
                patch_features_dir, f"{wsi.name}.h5"
            )  # Assumes h5 input
            slide_feature_file_path = os.path.join(
                slide_features_base_dir, f"{wsi.name}.{saveas}"
            )
            progress_bar.set_postfix_str(f"{slide_fullname}")

            # --- Pre-computation Checks ---
            if os.path.exists(slide_feature_file_path) and not is_locked(
                slide_feature_file_path
            ):
                update_log(
                    log_fp, slide_fullname, "DONE - Slide features already extracted"
                )
                self.cleanup(slide_fullname)
                continue
            if is_locked(slide_feature_file_path):
                update_log(log_fp, slide_fullname, "SKIP - Locked")
                continue
            if not os.path.exists(patch_feature_h5_path):
                update_log(
                    log_fp,
                    slide_fullname,
                    f"SKIP - Required patch feature file not found: {patch_feature_h5_path}",
                )
                continue
            # WSI file check is less critical here, but we might need WSI object for metadata
            wsi_current_path = (
                os.path.join(self.wsi_cache, slide_fullname)
                if self.wsi_cache
                else wsi.original_path
            )
            if not os.path.exists(wsi_current_path):
                logger.debug(
                    f"WSI file {wsi_current_path} missing, but proceeding with feature file."
                )
                # update_log(log_fp, slide_fullname, f"INFO - WSI file missing at {wsi_current_path}") # Optional logging

            # --- Perform Slide Feature Extraction ---
            try:
                create_lock(slide_feature_file_path)
                update_log(log_fp, slide_fullname, "LOCK - Extracting slide features")
                # Initialize WSI mainly for metadata access if needed by slide encoder
                wsi._lazy_initialize()

                generated_slide_feature_path = wsi.extract_slide_features(
                    patch_features_path=patch_feature_h5_path,
                    slide_encoder=slide_encoder,
                    save_features=slide_features_base_dir,  # Pass directory
                    device=device,
                    # saveas='h5' is implicit in wsi.extract_slide_features
                )

                if not os.path.exists(generated_slide_feature_path):
                    raise FileNotFoundError(
                        f"Slide feature file {generated_slide_feature_path} not created."
                    )
                update_log(log_fp, slide_fullname, "DONE - Slide features extracted")

            except Exception as e:
                error_msg = f"ERROR during slide feature extraction: {e}"
                update_log(log_fp, slide_fullname, error_msg)
                logger.error(
                    f"Error extracting slide features for {slide_fullname}: {e}"
                )
                if isinstance(e, KeyboardInterrupt):
                    print("Slide feature extraction interrupted.")
                    raise e
                if not self.skip_errors:
                    raise RuntimeError(
                        f"Error extracting slide features for {slide_fullname}: {e}"
                    ) from e
            finally:
                if os.path.exists(slide_feature_file_path + ".lock"):
                    try:
                        remove_lock(slide_feature_file_path)
                    except OSError as lock_err:
                        logger.warning(
                            f"Could not remove lock {slide_feature_file_path}.lock: {lock_err}"
                        )
                self.cleanup(slide_fullname)
                wsi.close()  # Close WSI handle

        logger.info(
            f"Slide feature extraction finished. Features in: {slide_features_base_dir}"
        )
        return slide_features_base_dir

    def cleanup(self, filename: str) -> None:
        """Removes the specified WSI file from the cache directory if enabled."""
        if self.wsi_cache and self.clear_cache:
            cache_file_path = os.path.join(self.wsi_cache, filename)
            if os.path.exists(cache_file_path):
                if not is_locked(cache_file_path):
                    try:
                        os.remove(cache_file_path)
                        logger.debug(f"Cleaned {filename} from cache.")
                        # update_log(os.path.join(self.wsi_cache, '_cache_log.txt'), filename, 'INFO - Cleaned from cache')
                    except OSError as e:
                        logger.warning(
                            f"Failed to remove {cache_file_path} from cache: {e}"
                        )
                        # update_log(os.path.join(self.wsi_cache, '_cache_log.txt'), filename, f'ERROR - Cleanup failed: {e}')
                # else: # Optional: Log skipped cleanup due to lock
                # logger.debug(f"Skipping cleanup of locked file: {cache_file_path}")
                # update_log(os.path.join(self.wsi_cache, '_cache_log.txt'), filename, 'INFO - Cleanup skipped (locked)')

    def save_config(
        self,
        saveto: PathLike,
        local_attrs: Optional[Dict[str, Any]] = None,
        ignore: Optional[List[str]] = None,
    ) -> None:
        """Saves the processor's configuration and job parameters to a JSON file."""
        if ignore is None:
            ignore = [
                "wsis",
                "loop",
                "wsis_source",
                "wsi_cache",
            ]  # Exclude sensitive/large/redundant

        config_to_save = {}

        # Add instance attributes (filter sensitive/large ones)
        for k, v in vars(self).items():
            if k not in ignore and not k.startswith("_"):  # Exclude private attrs
                try:
                    json.dumps(v, cls=JSONsaver)  # Quick check with custom saver
                    config_to_save[k] = v
                except (TypeError, OverflowError):
                    config_to_save[k] = (
                        f"<Object type: {type(v).__name__}>"  # Represent non-serializable
                    )

        # Add/overwrite with local attributes from the specific job method
        if local_attrs:
            for k, v in local_attrs.items():
                if k not in ignore:
                    try:
                        json.dumps(v, cls=JSONsaver)
                        config_to_save[k] = v
                    except (TypeError, OverflowError):
                        # Special handling for models - just save name if possible
                        if isinstance(v, torch.nn.Module) and hasattr(v, "enc_name"):
                            config_to_save[k] = (
                                f"<Model: {getattr(v, 'enc_name', type(v).__name__)}>"
                            )
                        elif isinstance(v, torch.nn.Module) and hasattr(
                            v, "model_name"
                        ):
                            config_to_save[k] = (
                                f"<Model: {getattr(v, 'model_name', type(v).__name__)}>"
                            )
                        else:
                            config_to_save[k] = f"<Object type: {type(v).__name__}>"

        # Ensure directory exists and save
        try:
            os.makedirs(os.path.dirname(saveto), exist_ok=True)
            with open(saveto, "w") as f:
                json.dump(config_to_save, f, indent=4, cls=JSONsaver)
            logger.debug(f"Configuration saved to {saveto}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {saveto}: {e}")
