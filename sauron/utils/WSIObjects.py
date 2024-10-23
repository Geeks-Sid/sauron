from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import geopandas as gpd
import numpy as np
import openslide
from PIL import Image

from sauron.utils.warnings import CuImageWarning


def is_cuimage_instance(image):
    try:
        from cucim import CuImage
    except ImportError:
        CuImage = None
        CuImageWarning.warn()
    return CuImage is not None and isinstance(image, CuImage)


class WholeSlideImage(ABC):
    def __init__(self, image_source):
        self.image_source = image_source

        if not (
            isinstance(image_source, openslide.OpenSlide)
            or isinstance(image_source, np.ndarray)
            or is_cuimage_instance(image_source)
        ):
            raise ValueError(f"Invalid image type: {type(image_source)}")

        self.width, self.height = self.get_dimensions()

    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_dimensions(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> np.ndarray:
        pass

    @abstractmethod
    def get_thumbnail(self, width: int, height: int) -> np.ndarray:
        pass

    def __repr__(self) -> str:
        return f"<width={self.width}, height={self.height}, backend={self.__class__.__name__}>"

    @abstractmethod
    def create_patcher(
        self,
        patch_size: int,
        src_mpp: float,
        dst_mpp: Optional[float] = None,
        overlap: int = 0,
        mask: Optional[gpd.GeoDataFrame] = None,
        coords_only: bool = False,
        custom_coords: Optional[np.ndarray] = None,
    ) -> "WSIPatcher":
        pass


def wsi_factory(image_source) -> WholeSlideImage:
    try:
        from cucim import CuImage
    except ImportError:
        CuImage = None
        CuImageWarning.warn()

    if isinstance(image_source, WholeSlideImage):
        return image_source

    image_type_map = {
        openslide.OpenSlide: OpenSlideWSI,
        np.ndarray: NumpyWSI,
        str: lambda src: (
            CuImageWSI(CuImage(src))
            if CuImage
            else OpenSlideWSI(openslide.OpenSlide(src))
        ),
    }

    for image_type, constructor in image_type_map.items():
        if isinstance(image_source, image_type):
            return constructor(image_source)

    if is_cuimage_instance(image_source):
        return CuImageWSI(image_source)

    raise ValueError(f"Unsupported image type: {type(image_source)}")


class NumpyWSI(WholeSlideImage):
    def __init__(self, image: np.ndarray):
        super().__init__(image)

    def to_numpy(self) -> np.ndarray:
        return self.image_source

    def get_dimensions(self) -> Tuple[int, int]:
        return self.image_source.shape[1], self.image_source.shape[0]

    def read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> np.ndarray:
        x_start, y_start = location
        x_size, y_size = size
        return self.image_source[y_start : y_start + y_size, x_start : x_start + x_size]

    def get_thumbnail(self, width: int, height: int) -> np.ndarray:
        return cv2.resize(self.image_source, (width, height))

    def create_patcher(
        self,
        patch_size: int,
        src_mpp: float,
        dst_mpp: Optional[float] = None,
        overlap: int = 0,
        mask: Optional[gpd.GeoDataFrame] = None,
        coords_only: bool = False,
        custom_coords: Optional[np.ndarray] = None,
    ) -> "NumpyWSIPatcher":
        return NumpyWSIPatcher(
            self,
            patch_size,
            src_mpp,
            dst_mpp,
            overlap,
            mask,
            coords_only,
            custom_coords,
        )


class OpenSlideWSI(WholeSlideImage):
    def __init__(self, image: openslide.OpenSlide):
        super().__init__(image)

    def to_numpy(self) -> np.ndarray:
        return self.get_thumbnail(self.width, self.height)

    def get_dimensions(self) -> Tuple[int, int]:
        return self.image_source.dimensions

    def read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> np.ndarray:
        return np.array(self.image_source.read_region(location, level, size))[:, :, :3]

    def get_thumbnail(self, width: int, height: int) -> np.ndarray:
        return np.array(self.image_source.get_thumbnail((width, height)))

    def get_best_level_for_downsample(self, downsample: float) -> int:
        return self.image_source.get_best_level_for_downsample(downsample)

    def level_dimensions(self) -> List[Tuple[int, int]]:
        return self.image_source.level_dimensions

    def level_downsamples(self) -> List[float]:
        return self.image_source.level_downsamples

    def create_patcher(
        self,
        patch_size: int,
        src_mpp: float,
        dst_mpp: Optional[float] = None,
        overlap: int = 0,
        mask: Optional[gpd.GeoDataFrame] = None,
        coords_only: bool = False,
        custom_coords: Optional[np.ndarray] = None,
    ) -> "OpenSlideWSIPatcher":
        return OpenSlideWSIPatcher(
            self,
            patch_size,
            src_mpp,
            dst_mpp,
            overlap,
            mask,
            coords_only,
            custom_coords,
        )


class CuImageWSI(WholeSlideImage):
    def __init__(self, image):
        super().__init__(image)

    def to_numpy(self) -> np.ndarray:
        return self.get_thumbnail(self.width, self.height)

    def get_dimensions(self) -> Tuple[int, int]:
        return self.image_source.resolutions["level_dimensions"][0]

    def read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> np.ndarray:
        return np.array(
            self.image_source.read_region(location=location, level=level, size=size)
        )[:, :, :3]

    def get_thumbnail(self, width: int, height: int) -> np.ndarray:
        downsample = self.width / width
        level = self.get_best_level_for_downsample(downsample)
        curr_width, curr_height = self.image_source.resolutions["level_dimensions"][
            level
        ]
        thumbnail = np.array(
            self.image_source.read_region(
                location=(0, 0), level=level, size=(curr_width, curr_height)
            )
        )[:, :, :3]
        return cv2.resize(thumbnail, (width, height))

    def get_best_level_for_downsample(self, downsample: float) -> int:
        downsamples = self.image_source.resolutions["level_downsamples"]
        for i, level_downsample in enumerate(downsamples):
            if downsample < level_downsample:
                return max(i - 1, 0)
        return len(downsamples) - 1

    def level_dimensions(self) -> List[Tuple[int, int]]:
        return self.image_source.resolutions["level_dimensions"]

    def level_downsamples(self) -> List[float]:
        return self.image_source.resolutions["level_downsamples"]

    def create_patcher(
        self,
        patch_size: int,
        src_mpp: float,
        dst_mpp: Optional[float] = None,
        overlap: int = 0,
        mask: Optional[gpd.GeoDataFrame] = None,
        coords_only: bool = False,
        custom_coords: Optional[np.ndarray] = None,
    ) -> "CuImageWSIPatcher":
        return CuImageWSIPatcher(
            self,
            patch_size,
            src_mpp,
            dst_mpp,
            overlap,
            mask,
            coords_only,
            custom_coords,
        )


class WSIPatcher(ABC):
    """Iterator class to handle patch extraction, scaling, and mask intersection."""

    def __init__(
        self,
        wsi: WholeSlideImage,
        patch_size: int,
        src_mpp: float,
        dst_mpp: Optional[float] = None,
        overlap: int = 0,
        mask: Optional[gpd.GeoDataFrame] = None,
        coords_only: bool = False,
        custom_coords: Optional[np.ndarray] = None,
    ):
        self.wsi = wsi
        self.overlap = overlap
        self.width, self.height = self.wsi.get_dimensions()
        self.patch_size_target = patch_size
        self.mask = mask
        self.coords_only = coords_only
        self.custom_coords = custom_coords
        self.current_index = 0

        self.downsample = 1.0 if dst_mpp is None else dst_mpp / src_mpp
        self.patch_size_src = round(patch_size * self.downsample)

        self.level, self.patch_size_level, self.overlap_level = self._prepare_patching()

        if custom_coords is None:
            self.cols, self.rows = self._calculate_cols_rows()
            grid_coords = np.array(
                [[col, row] for col in range(self.cols) for row in range(self.rows)]
            )
            self.all_coords = np.array(
                [self._grid_to_coordinates(col, row) for col, row in grid_coords]
            )
        else:
            self.all_coords = custom_coords

        if self.mask is not None:
            self.valid_coords = self._filter_coords_with_mask(self.all_coords)
        else:
            self.valid_coords = self.all_coords

    def _colrow_to_xy(self, col, row):
        """Convert col row of a tile to its top-left coordinates before rescaling (x, y)"""
        offset = self.patch_size_src - self.overlap
        x = col * offset
        y = row * offset
        return x, y

    def _compute_masked(self, coords) -> None:
        """Compute tiles which any corner falls under the tissue"""

        # Filter coordinates by bounding boxes of mask polygons
        patch_size_offset = self.patch_size_src
        bounding_boxes = self.mask.geometry.bounds
        valid_coords = [
            coords[
                (coords[:, 0] >= bbox["minx"] - patch_size_offset)
                & (coords[:, 0] <= bbox["maxx"] + patch_size_offset)
                & (coords[:, 1] >= bbox["miny"] - patch_size_offset)
                & (coords[:, 1] <= bbox["maxy"] + patch_size_offset)
            ]
            for _, bbox in bounding_boxes.iterrows()
        ]

        if valid_coords:
            coords = np.unique(np.vstack(valid_coords), axis=0)
        else:
            return 0, np.array([])

        # Calculate corner coordinates
        corner_offsets = np.array(
            [
                [0, 0],
                [patch_size_offset, 0],
                [0, patch_size_offset],
                [patch_size_offset, patch_size_offset],
            ]
        )
        corners = (coords[:, None, :] + corner_offsets).reshape(-1, 2)

        union_mask = self.mask.union_all()

        # Check if any of the corners fall within the mask
        points = gpd.points_from_xy(corners[:, 0], corners[:, 1])
        valid_mask = (
            gpd.GeoSeries(points).within(union_mask).values.reshape(-1, 4).any(axis=1)
        )
        valid_patches_nb = valid_mask.sum()
        valid_coords = coords[valid_mask]

        return valid_patches_nb, valid_coords

    def __len__(self):
        return len(self.valid_coords)

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self):
            raise StopIteration
        item = self[self.current_index]
        self.current_index += 1
        return item

    def __getitem__(self, index):
        if 0 <= index < len(self):
            x, y = self.valid_coords[index]
            if self.coords_only:
                return x, y
            return self.get_patch_at(x, y)
        else:
            raise IndexError("Index out of range")

    def _grid_to_coordinates(self, col: int, row: int) -> Tuple[int, int]:
        x = col * self.patch_size_src - self.overlap * max(col - 1, 0)
        y = row * self.patch_size_src - self.overlap * max(row - 1, 0)
        return x, y

    def _calculate_cols_rows(self) -> Tuple[int, int]:
        cols = (self.width + self.patch_size_src - 1) // (
            self.patch_size_src - self.overlap
        )
        rows = (self.height + self.patch_size_src - 1) // (
            self.patch_size_src - self.overlap
        )
        return cols, rows

    def _filter_coords_with_mask(self, coords: np.ndarray) -> np.ndarray:
        union_mask = self.mask.unary_union
        patches_polygons = [
            gpd.box(x, y, x + self.patch_size_src, y + self.patch_size_src)
            for x, y in coords
        ]
        patches_gdf = gpd.GeoDataFrame(geometry=patches_polygons)
        intersects = patches_gdf.intersects(union_mask)
        return coords[intersects.values]

    def get_patch_at(self, x: int, y: int) -> Tuple[np.ndarray, int, int]:
        patch_image = self.wsi.read_region(
            location=(x, y),
            level=self.level,
            size=(self.patch_size_level, self.patch_size_level),
        )
        if self.patch_size_target != self.patch_size_level:
            patch_image = cv2.resize(
                patch_image, (self.patch_size_target, self.patch_size_target)
            )
        return patch_image[:, :, :3], x, y

    def get_tile_xy(self, x: int, y: int) -> Tuple[np.ndarray, int, int]:
        raw_tile = self.wsi.read_region(
            location=(x, y),
            level=self.level,
            size=(self.patch_size_level, self.patch_size_level),
        )
        tile = np.array(raw_tile)
        if self.patch_size_target is not None:
            tile = cv2.resize(tile, (self.patch_size_target, self.patch_size_target))
        assert x < self.width and y < self.height
        return tile[:, :, :3], x, y

    def get_tile(self, col: int, row: int) -> Tuple[np.ndarray, int, int]:
        """Get tile at position (column, row)

        Args:
            col (int): column
            row (int): row

        Returns:
            Tuple[np.ndarray, int, int]: (tile, pixel x of top-left corner (before rescaling), pixel_y of top-left corner (before rescaling))
        """
        if self.custom_coords is not None:
            raise ValueError(
                "Can't use get_tile as 'custom_coords' was passed to the constructor"
            )

        x, y = self._colrow_to_xy(col, row)
        return self.get_tile_xy(x, y)

    def _compute_cols_rows(self) -> Tuple[int, int]:
        col = 0
        row = 0
        x, y = self._colrow_to_xy(col, row)
        while x < self.width:
            col += 1
            x, _ = self._colrow_to_xy(col, row)
        cols = col
        while y < self.height:
            row += 1
            _, y = self._colrow_to_xy(col, row)
        rows = row
        return cols, rows

    def save_visualization(self, path: str, vis_width: int = 1000, dpi: int = 150):
        visualization = generate_visualization(
            self.wsi,
            self.mask,
            self.valid_coords,
            line_color=(0, 255, 0),
            line_thickness=2,
            target_width=vis_width,
        )
        visualization.save(path, dpi=(dpi, dpi))


class OpenSlideWSIPatcher(WSIPatcher):
    def _prepare_patching(self) -> Tuple[int, int, int]:
        level = self.wsi.get_best_level_for_downsample(self.downsample)
        level_downsample = self.wsi.level_downsamples()[level]
        patch_size_level = round(self.patch_size_src / level_downsample)
        overlap_level = round(self.overlap / level_downsample)
        return level, patch_size_level, overlap_level


class CuImageWSIPatcher(WSIPatcher):
    def _prepare_patching(self) -> Tuple[int, int, int]:
        level = self.wsi.get_best_level_for_downsample(self.downsample)
        level_downsample = self.wsi.level_downsamples()[level]
        patch_size_level = round(self.patch_size_src / level_downsample)
        overlap_level = round(self.overlap / level_downsample)
        return level, patch_size_level, overlap_level


class NumpyWSIPatcher(WSIPatcher):
    def _prepare_patching(self) -> Tuple[int, int, int]:
        patch_size_level = self.patch_size_src
        overlap_level = self.overlap
        level = -1  # Not applicable for numpy arrays
        return level, patch_size_level, overlap_level


def draw_contours_on_image(
    contours: gpd.GeoDataFrame,
    image: np.ndarray,
    line_color: Tuple[int, int, int] = (0, 255, 0),
    line_thickness: int = 1,
    downsample_factor: float = 1.0,
) -> np.ndarray:
    for _, row in contours.iterrows():
        exterior_coords = np.array(
            [
                [int(x * downsample_factor), int(y * downsample_factor)]
                for x, y in row.geometry.exterior.coords
            ]
        )
        interiors = [
            np.array(
                [
                    [int(x * downsample_factor), int(y * downsample_factor)]
                    for x, y in interior.coords
                ]
            )
            for interior in row.geometry.interiors
        ]
        cv2.drawContours(
            image, [exterior_coords], -1, line_color, line_thickness, cv2.LINE_8
        )
        for hole in interiors:
            cv2.drawContours(image, [hole], -1, (0, 0, 0), cv2.FILLED, cv2.LINE_8)
    return image


def generate_visualization(
    wsi: WholeSlideImage,
    tissue_contours: Optional[gpd.GeoDataFrame],
    patch_coords: np.ndarray,
    line_color: Tuple[int, int, int] = (0, 255, 0),
    line_thickness: int = 2,
    target_width: int = 1000,
) -> Image:
    width, height = wsi.get_dimensions()
    downsample_factor = target_width / width

    thumbnail = wsi.get_thumbnail(
        int(width * downsample_factor), int(height * downsample_factor)
    )
    overlay = np.zeros_like(thumbnail)

    if tissue_contours is not None:
        downsampled_contours = tissue_contours.copy()
        draw_contours_on_image(
            downsampled_contours, overlay, line_color, line_thickness, downsample_factor
        )

    for x, y in patch_coords:
        x_ds, y_ds = int(x * downsample_factor), int(y * downsample_factor)
        ps_ds = int(wsi.patch_size_src * downsample_factor)
        cv2.rectangle(
            overlay,
            (x_ds, y_ds),
            (x_ds + ps_ds, y_ds + ps_ds),
            (255, 0, 0),
            line_thickness,
        )

    alpha = 0.4
    combined_image = cv2.addWeighted(thumbnail, 1 - alpha, overlay, alpha, 0)
    return Image.fromarray(combined_image.astype(np.uint8))


def get_pixel_spacing(slide: openslide.OpenSlide) -> float:
    """
    Extracts the pixel spacing (in micrometers per pixel) from a whole slide image.

    Parameters:
        slide (openslide.OpenSlide): The slide object.

    Returns:
        float: Pixel spacing in micrometers.
    """
    try:
        pixel_spacing = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        if pixel_spacing == 0:
            raise ValueError("Pixel spacing information is zero in the slide metadata.")
        return pixel_spacing
    except KeyError:
        raise ValueError("Pixel spacing information is not available in the slide.")
    except Exception as e:
        raise ValueError(f"Could not retrieve pixel spacing: {e}")
