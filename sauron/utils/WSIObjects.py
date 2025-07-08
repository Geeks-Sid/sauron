from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cucim
import cv2
import geopandas as gpd
import numpy as np
import openslide
from PIL import Image

from sauron.utils.warnings import CuImageWarning


def is_cuimage_instance(image: object) -> bool:
    """Checks if a given object is an instance of cucim.CuImage.

    This function safely checks for the CuImage type without raising an
    ImportError if cuCIM is not installed. A warning is issued if the
    library is not found.

    Args:
        image: The object to check.

    Returns:
        True if the object is a CuImage instance, False otherwise.
    """
    try:
        from cucim import CuImage
    except ImportError:
        CuImage = None
        CuImageWarning.warn()
    return CuImage is not None and isinstance(image, CuImage)


class WholeSlideImage(ABC):
    """Abstract base class for a unified whole-slide image (WSI) interface.

    This class defines a standard set of methods for interacting with
    whole-slide images, regardless of the underlying backend (e.g., OpenSlide,
    NumPy array, cuCIM). It ensures that different WSI formats can be handled
    interchangeably.

    Attributes:
        image_source: The underlying image object (e.g., openslide.OpenSlide).
        width (int): The width of the level 0 image in pixels.
        height (int): The height of the level 0 image in pixels.
    """

    def __init__(self, image_source: object):
        """Initializes the WholeSlideImage.

        Args:
            image_source: The source of the image data. Supported types are
                determined by subclasses and the wsi_factory.

        Raises:
            ValueError: If the image_source type is not supported.
        """
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
        """Converts the entire WSI to a NumPy array.

        Note: This can consume a large amount of memory for high-resolution images.
        It's often used for smaller images or for getting a full-resolution
        view when memory is not a concern.

        Returns:
            A NumPy array representing the full image.
        """
        pass

    @abstractmethod
    def get_dimensions(self) -> Tuple[int, int]:
        """Gets the dimensions of the level 0 image.

        Returns:
            A tuple containing the (width, height) of the slide in pixels.
        """
        pass

    @abstractmethod
    def read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> np.ndarray:
        """Reads a specified region from the slide.

        Args:
            location: A tuple (x, y) with the top-left coordinates of the region
                at level 0.
            level: The resolution level to read from.
            size: A tuple (width, height) of the region to read at the specified
                level.

        Returns:
            A NumPy array representing the image region.
        """
        pass

    @abstractmethod
    def get_thumbnail(self, width: int, height: int) -> np.ndarray:
        """Generates a thumbnail of the WSI.

        Args:
            width: The desired width of the thumbnail.
            height: The desired height of the thumbnail.

        Returns:
            A NumPy array representing the resized thumbnail.
        """
        pass

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation of the WSI."""
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
        """Creates a patcher instance for iterating over image patches.

        Args:
            patch_size: The desired size of the output patch in pixels.
            src_mpp: The microns-per-pixel (MPP) of the source WSI.
            dst_mpp: The desired MPP of the output patches. If None, no rescaling
                is performed.
            overlap: The number of overlapping pixels between adjacent patches.
            mask: A GeoDataFrame containing polygons to filter patches. Only
                patches that intersect with the mask are yielded.
            coords_only: If True, the patcher yields only (x, y) coordinates
                instead of image data.
            custom_coords: A NumPy array of (x, y) coordinates to extract patches
                from, bypassing the default grid generation.

        Returns:
            An instance of a WSIPatcher subclass.
        """
        pass


def wsi_factory(image_source: object) -> WholeSlideImage:
    """Factory function to create a WholeSlideImage instance from various sources.

    This function automatically selects the appropriate WSI backend based on the
    type of the input `image_source`. It supports file paths (strings),
    OpenSlide objects, NumPy arrays, and cuCIM objects.

    Args:
        image_source: The image source. Can be a file path (str),
            `openslide.OpenSlide`, `np.ndarray`, `cucim.CuImage`, or another
            `WholeSlideImage` instance.

    Returns:
        An appropriate subclass of WholeSlideImage.

    Raises:
        ValueError: If the `image_source` type is not supported.
    """
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
    """A WholeSlideImage implementation for NumPy arrays."""

    def __init__(self, image: np.ndarray):
        """Initializes the NumpyWSI.

        Args:
            image: The NumPy array representing the image.
        """
        super().__init__(image)

    def to_numpy(self) -> np.ndarray:
        """Returns the underlying NumPy array."""
        return self.image_source

    def get_dimensions(self) -> Tuple[int, int]:
        """Returns the dimensions (width, height) of the NumPy array."""
        return self.image_source.shape[1], self.image_source.shape[0]

    def read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> np.ndarray:
        """Reads a region from the NumPy array.

        Args:
            location: Top-left (x, y) coordinates of the region.
            level: Ignored for NumPy arrays (always level 0).
            size: The (width, height) of the region to extract.

        Returns:
            A NumPy array view of the specified region.
        """
        x_start, y_start = location
        x_size, y_size = size
        return self.image_source[y_start : y_start + y_size, x_start : x_start + x_size]

    def get_thumbnail(self, width: int, height: int) -> np.ndarray:
        """Creates a resized thumbnail from the NumPy array."""
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
        """Creates a patcher for the NumPy array."""
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
    """A WholeSlideImage implementation for OpenSlide-compatible files."""

    def __init__(self, image: openslide.OpenSlide):
        """Initializes the OpenSlideWSI.

        Args:
            image: An initialized openslide.OpenSlide object.
        """
        super().__init__(image)

    def to_numpy(self) -> np.ndarray:
        """Returns the entire image as a NumPy array by creating a full-size thumbnail."""
        return self.get_thumbnail(self.width, self.height)

    def get_dimensions(self) -> Tuple[int, int]:
        """Returns the level 0 dimensions from the OpenSlide object."""
        return self.image_source.dimensions

    def read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> np.ndarray:
        """Reads a region using the OpenSlide backend."""
        return np.array(self.image_source.read_region(location, level, size))[:, :, :3]

    def get_thumbnail(self, width: int, height: int) -> np.ndarray:
        """Gets a thumbnail using the OpenSlide backend."""
        return np.array(self.image_source.get_thumbnail((width, height)))

    def get_best_level_for_downsample(self, downsample: float) -> int:
        """Determines the best WSI level for a given downsample factor."""
        return self.image_source.get_best_level_for_downsample(downsample)

    def level_dimensions(self) -> List[Tuple[int, int]]:
        """Gets the dimensions of each level in the WSI."""
        return self.image_source.level_dimensions

    def level_downsamples(self) -> List[float]:
        """Gets the downsample factor for each level in the WSI."""
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
        """Creates a patcher for the OpenSlide WSI."""
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
    """A WholeSlideImage implementation for cuCIM-compatible files."""

    def __init__(self, image: "cucim.CuImage"):
        """Initializes the CuImageWSI.

        Args:
            image: An initialized cucim.CuImage object.
        """
        super().__init__(image)

    def to_numpy(self) -> np.ndarray:
        """Returns the entire image as a NumPy array by creating a full-size thumbnail."""
        return self.get_thumbnail(self.width, self.height)

    def get_dimensions(self) -> Tuple[int, int]:
        """Returns the level 0 dimensions from the cuCIM object."""
        return self.image_source.resolutions["level_dimensions"][0]

    def read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> np.ndarray:
        """Reads a region using the cuCIM backend."""
        return np.array(
            self.image_source.read_region(location=location, level=level, size=size)
        )[:, :, :3]

    def get_thumbnail(self, width: int, height: int) -> np.ndarray:
        """Gets a thumbnail using the cuCIM backend.

        This method selects the most appropriate resolution level to read from
        and then resizes to the target dimensions.
        """
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
        """Determines the best WSI level for a given downsample factor."""
        downsamples = self.image_source.resolutions["level_downsamples"]
        for i, level_downsample in enumerate(downsamples):
            if downsample < level_downsample:
                return max(i - 1, 0)
        return len(downsamples) - 1

    def level_dimensions(self) -> List[Tuple[int, int]]:
        """Gets the dimensions of each level in the WSI."""
        return self.image_source.resolutions["level_dimensions"]

    def level_downsamples(self) -> List[float]:
        """Gets the downsample factor for each level in the WSI."""
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
        """Creates a patcher for the cuCIM WSI."""
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
    """Iterator class to handle patch extraction, scaling, and mask intersection.

    This class provides an iterable interface to efficiently extract patches from a
    WholeSlideImage. It manages grid generation, overlap, scaling based on MPP,
    and filtering based on a supplied geometry mask.

    This is an abstract base class; concrete implementations must provide the
    _prepare_patching method.

    Attributes:
        wsi (WholeSlideImage): The WSI object to patch.
        patch_size_target (int): The final size of the patches after any resizing.
        patch_size_src (int): The size of the patch to read from the source WSI at
            level 0 resolution.
        downsample (float): The calculated downsampling factor.
        level (int): The optimal WSI level to read from.
        valid_coords (np.ndarray): An array of (x, y) coordinates for the
            patches that will be yielded by the iterator.
    """

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
        """Initializes the WSIPatcher.

        Args:
            wsi: The WholeSlideImage object to process.
            patch_size: The desired output size of each patch in pixels.
            src_mpp: The microns-per-pixel (MPP) of the source WSI.
            dst_mpp: The desired MPP for the output patches. If provided, patches
                are read at a higher resolution and downscaled. If None,
                no rescaling occurs.
            overlap: The number of overlapping pixels between adjacent patches.
            mask: A GeoDataFrame containing polygon geometries. Only patches that
                intersect with these geometries will be processed.
            coords_only: If True, the iterator yields only (x, y) coordinates
                instead of the full patch data.
            custom_coords: An optional NumPy array of (x, y) coordinates to
                extract patches from, bypassing the default grid generation.
        """
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

    @abstractmethod
    def _prepare_patching(self) -> Tuple[int, int, int]:
        """Prepares patching parameters specific to the WSI backend.

        This method must be implemented by subclasses to calculate the optimal
        WSI level and the corresponding patch and overlap sizes at that level.

        Returns:
            A tuple of (level, patch_size_level, overlap_level).
        """
        pass

    def _filter_coords_with_mask(self, coords: np.ndarray) -> np.ndarray:
        """Filters patch coordinates to keep only those intersecting the mask."""
        union_mask = self.mask.unary_union
        patch_polygons = [
            gpd.box(x, y, x + self.patch_size_src, y + self.patch_size_src)
            for x, y in coords
        ]
        patches_gdf = gpd.GeoDataFrame(geometry=patch_polygons)
        intersects = patches_gdf.intersects(union_mask)
        return coords[intersects.values]

    def __len__(self) -> int:
        """Returns the total number of valid patches."""
        return len(self.valid_coords)

    def __iter__(self) -> "WSIPatcher":
        """Returns the iterator object itself."""
        self.current_index = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, int, int] | Tuple[int, int]:
        """Returns the next patch or coordinate."""
        if self.current_index >= len(self):
            raise StopIteration
        item = self[self.current_index]
        self.current_index += 1
        return item

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, int] | Tuple[int, int]:
        """Gets a patch or coordinate by its index.

        Args:
            index: The index of the valid patch coordinate.

        Returns:
            If `coords_only` is False, returns a tuple of (patch_image, x, y).
            If `coords_only` is True, returns a tuple of (x, y).

        Raises:
            IndexError: If the index is out of range.
        """
        if 0 <= index < len(self):
            x, y = self.valid_coords[index]
            if self.coords_only:
                return x, y
            return self.get_patch_at(x, y)
        else:
            raise IndexError("Index out of range")

    def _grid_to_coordinates(self, col: int, row: int) -> Tuple[int, int]:
        """Converts grid indices (col, row) to pixel coordinates (x, y)."""
        x = col * self.patch_size_src - self.overlap * max(col - 1, 0)
        y = row * self.patch_size_src - self.overlap * max(row - 1, 0)
        return x, y

    def _calculate_cols_rows(self) -> Tuple[int, int]:
        """Calculates the number of columns and rows in the patch grid."""
        cols = (self.width + self.patch_size_src - 1) // (
            self.patch_size_src - self.overlap
        )
        rows = (self.height + self.patch_size_src - 1) // (
            self.patch_size_src - self.overlap
        )
        return cols, rows

    def get_patch_at(self, x: int, y: int) -> Tuple[np.ndarray, int, int]:
        """Reads and resizes a single patch at the given level 0 coordinates.

        Args:
            x: The level 0 x-coordinate of the top-left corner.
            y: The level 0 y-coordinate of the top-left corner.

        Returns:
            A tuple containing (patch_image, x, y). The patch image is a
            NumPy array resized to `patch_size_target`.
        """
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

    def save_visualization(self, path: str, vis_width: int = 1000, dpi: int = 150):
        """Saves a visualization of the WSI with patch locations and masks.

        Args:
            path: The file path to save the visualization image.
            vis_width: The target width of the visualization image.
            dpi: The dots-per-inch for the saved image.
        """
        visualization = generate_visualization(
            self,  # Pass the patcher instance itself
            self.wsi,
            self.mask,
            self.valid_coords,
            line_color=(0, 255, 0),
            line_thickness=2,
            target_width=vis_width,
        )
        visualization.save(path, dpi=(dpi, dpi))


class OpenSlideWSIPatcher(WSIPatcher):
    """A WSIPatcher implementation for OpenSlide-backed WSIs."""

    def _prepare_patching(self) -> Tuple[int, int, int]:
        """Calculates patching parameters using OpenSlide's level metadata."""
        level = self.wsi.get_best_level_for_downsample(self.downsample)
        level_downsample = self.wsi.level_downsamples()[level]
        patch_size_level = round(self.patch_size_src / level_downsample)
        overlap_level = round(self.overlap / level_downsample)
        return level, patch_size_level, overlap_level


class CuImageWSIPatcher(WSIPatcher):
    """A WSIPatcher implementation for cuCIM-backed WSIs."""

    def _prepare_patching(self) -> Tuple[int, int, int]:
        """Calculates patching parameters using cuCIM's level metadata."""
        level = self.wsi.get_best_level_for_downsample(self.downsample)
        level_downsample = self.wsi.level_downsamples()[level]
        patch_size_level = round(self.patch_size_src / level_downsample)
        overlap_level = round(self.overlap / level_downsample)
        return level, patch_size_level, overlap_level


class NumpyWSIPatcher(WSIPatcher):
    """A WSIPatcher implementation for NumPy array-backed WSIs."""

    def _prepare_patching(self) -> Tuple[int, int, int]:
        """Sets patching parameters for a NumPy array.

        Since NumPy arrays are single-resolution, level is set to -1 (N/A) and
        sizes are not scaled by a level downsample factor.
        """
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
    """Draws polygon contours from a GeoDataFrame onto a NumPy image.

    Args:
        contours: A GeoDataFrame with a 'geometry' column of Polygons.
        image: The NumPy array image on which to draw.
        line_color: The BGR color tuple for the contour lines.
        line_thickness: The thickness of the contour lines.
        downsample_factor: A factor to scale the contour coordinates to match
            the image's resolution.

    Returns:
        The image with contours drawn on it.
    """
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
    patcher: WSIPatcher,
    wsi: WholeSlideImage,
    tissue_contours: Optional[gpd.GeoDataFrame],
    patch_coords: np.ndarray,
    line_color: Tuple[int, int, int] = (0, 255, 0),
    line_thickness: int = 2,
    target_width: int = 1000,
) -> Image:
    """Generates a visualization image with WSI thumbnail, contours, and patches.

    Args:
        patcher: The WSIPatcher instance used for tiling. It provides patch size info.
        wsi: The WholeSlideImage object.
        tissue_contours: An optional GeoDataFrame with tissue polygons to draw.
        patch_coords: A NumPy array of (x, y) coordinates for the patches to draw.
        line_color: The color for the tissue contour lines.
        line_thickness: The line thickness for contours and patch rectangles.
        target_width: The desired width of the output visualization thumbnail.

    Returns:
        A PIL Image object of the combined visualization.
    """
    width, height = wsi.get_dimensions()
    downsample_factor = target_width / width

    thumbnail = wsi.get_thumbnail(
        int(width * downsample_factor), int(height * downsample_factor)
    )
    overlay = np.zeros_like(thumbnail)

    if tissue_contours is not None:
        draw_contours_on_image(
            tissue_contours, overlay, line_color, line_thickness, downsample_factor
        )

    for x, y in patch_coords:
        x_ds, y_ds = int(x * downsample_factor), int(y * downsample_factor)
        ps_ds = int(patcher.patch_size_src * downsample_factor)
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
    """Extracts the pixel spacing (in micrometers per pixel) from a whole slide image.

    Args:
        slide: The slide object.

    Returns:
        Pixel spacing in micrometers.

    Raises:
        ValueError: If pixel spacing information is missing, zero, or cannot
            be retrieved from the slide metadata.
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
