# sauron/feature_extraction/wsi/factory.py
import os
from typing import Literal, Optional, Union

from .cucim import CuCIMWSI
from .image import ImageWSI
from .openslide import OpenSlideWSI

WSIReaderType = Literal["openslide", "image", "cucim"]
OPENSLIDE_EXTENSIONS = {
    ".svs",
    ".tif",
    ".tiff",
    ".ndpi",
    ".vms",
    ".vmu",
    ".scn",
    ".mrxs",
}
CUCIM_EXTENSIONS = {".svs", ".tif", ".tiff"}  # Add other cucim formats if needed


def load_wsi(
    slide_path: str, reader_type: Optional[WSIReaderType] = None, **kwargs
) -> Union[OpenSlideWSI, ImageWSI, CuCIMWSI]:
    """
    Load a whole-slide image (WSI) using the appropriate backend.
    """
    ext = os.path.splitext(slide_path)[1].lower()

    if reader_type == "openslide":
        return OpenSlideWSI(slide_path=slide_path, **kwargs)

    elif reader_type == "image":
        return ImageWSI(slide_path=slide_path, **kwargs)

    elif reader_type == "cucim":
        if ext in CUCIM_EXTENSIONS:
            return CuCIMWSI(slide_path=slide_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format '{ext}' for CuCIM.")

    elif reader_type is None:  # Auto-detection
        if ext in OPENSLIDE_EXTENSIONS:
            return OpenSlideWSI(slide_path=slide_path, **kwargs)
        else:
            return ImageWSI(slide_path=slide_path, **kwargs)

    else:
        raise ValueError(
            f"Unknown reader_type: {reader_type}. Choose from 'openslide', 'image', or 'cucim'."
        )
