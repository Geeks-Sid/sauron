from typing import Dict, List, Optional, Tuple

import cv2
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if image.shape[:2] != mask.shape:
        raise ValueError("Image and mask dimensions do not match.")

    masked_image = np.where(mask[..., None], image, 255)
    return masked_image.astype(np.uint8)


def filter_and_group_contours(
    contours: List[np.ndarray],
    hierarchy: np.ndarray,
    filter_params: Dict[str, float],
    scaling_factor: float,
    pixel_size: float,
) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
    foreground_indices: List[int] = []
    hole_indices_per_contour: List[List[int]] = []
    top_level_indices = np.where(hierarchy[:, 1] == -1)[0] if hierarchy.size > 0 else []

    for idx in top_level_indices:
        contour = contours[idx]
        hole_indices = np.where(hierarchy[:, 1] == idx)[0]
        area = cv2.contourArea(contour) - sum(
            cv2.contourArea(contours[hole_idx]) for hole_idx in hole_indices
        )
        area *= (pixel_size**2) / (scaling_factor**2)

        if area > filter_params["min_area"]:
            foreground_indices.append(idx)
            hole_indices_per_contour.append(
                [
                    hole_idx
                    for hole_idx in hole_indices
                    if cv2.contourArea(contours[hole_idx]) * pixel_size**2
                    > filter_params["min_hole_area"]
                ]
            )

    foreground_contours = [contours[idx] for idx in foreground_indices]
    hole_contours = [
        [contours[hole_idx] for hole_idx in holes] for holes in hole_indices_per_contour
    ]

    return foreground_contours, hole_contours


def mask_to_geodataframe(
    mask: np.ndarray,
    keep_ids: Optional[List[int]] = None,
    exclude_ids: Optional[List[int]] = None,
    max_holes: int = 0,
    min_contour_area: int = 1000,
    pixel_size: float = 1.0,
    contour_scale: float = 1.0,
) -> gpd.GeoDataFrame:
    TARGET_SIZE = 2000
    scaling_factor = TARGET_SIZE / mask.shape[0]
    resized_mask = cv2.resize(
        mask, (int(mask.shape[1] * scaling_factor), int(mask.shape[0] * scaling_factor))
    )

    contours, hierarchy = cv2.findContours(
        resized_mask,
        cv2.RETR_TREE if max_holes == 0 else cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE,
    )
    hierarchy = (
        hierarchy.squeeze(axis=0)[:, 2:] if hierarchy is not None else np.array([])
    )

    filter_params: Dict[str, float] = {
        "min_area": min_contour_area * (pixel_size**2),
        "min_hole_area": 4000 * (pixel_size**2),
    }

    foreground_contours, hole_contours = filter_and_group_contours(
        contours, hierarchy, filter_params, scaling_factor, pixel_size
    )

    if not foreground_contours:
        raise ValueError("No contours detected.")

    # Scale contours back to original size
    scaled_foreground_contours = [
        (contour * (contour_scale / scaling_factor)).astype(int)
        for contour in foreground_contours
    ]
    scaled_hole_contours = [
        [(hole * (contour_scale / scaling_factor)).astype(int) for hole in holes]
        for holes in hole_contours
    ]

    contour_indices = (
        set(keep_ids) - set(exclude_ids)
        if keep_ids
        else set(range(len(scaled_foreground_contours))) - set(exclude_ids or [])
    )

    polygons: List[Polygon] = []
    for idx in contour_indices:
        holes = (
            [hole.squeeze(1) for hole in scaled_hole_contours[idx]]
            if scaled_hole_contours[idx]
            else None
        )
        exterior = scaled_foreground_contours[idx].squeeze(1)
        polygon = Polygon(exterior, holes)
        if not polygon.is_valid:
            polygon = fix_invalid_polygon(polygon)
        polygons.append(polygon)

    return gpd.GeoDataFrame({"tissue_id": list(contour_indices)}, geometry=polygons)


def fix_invalid_polygon(polygon: Polygon) -> Polygon:
    for buffer_value in [0, 0.1, -0.1, 0.2]:
        new_polygon = polygon.buffer(buffer_value)
        if new_polygon.is_valid:
            return new_polygon
    raise ValueError("Failed to create a valid polygon")
