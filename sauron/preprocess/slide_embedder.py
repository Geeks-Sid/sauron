import logging
import os
from pathlib import Path
from typing import List, Tuple

import openslide
from termcolor import colored
from tqdm import tqdm

# Assuming these are the correct locations and types for the sauron library
from sauron.preprocess.compute_embeddings import TileEmbedder
from sauron.utils.filehandler import PatientFolder
from sauron.utils.segmentation import TissueSegmenter
from sauron.utils.WSIObjects import OpenSlideWSI, get_pixel_size

# It's good practice to import GeoDataFrame if you are using it for type hints
# from geopandas import GeoDataFrame

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SlideEmbedder:
    """Orchestrates the processing of whole-slide images (WSIs) to generate tile embeddings."""

    def __init__(
        self,
        slide_dir: Path | str,
        output_dir: Path | str,
        magnification: int,
        patch_size: int,
        batch_size: int,
        encoder: str,
    ):
        """
        Initializes the SlideEmbedder.

        Args:
            slide_dir (Path | str): Directory containing the WSI files.
            output_dir (Path | str): Root directory where processing outputs will be saved.
            magnification (int): The target magnification level (e.g., 20 for 20x).
            patch_size (int): The size of the square patches to extract, in pixels.
            batch_size (int): The batch size for processing tiles.
        """
        # IMPROVEMENT: Use pathlib.Path for modern, object-oriented path handling.
        self.slide_dir = Path(slide_dir)
        self.output_dir = Path(output_dir)
        self.magnification = magnification
        self.patch_size = patch_size
        self.batch_size = batch_size

    def process_slides(self) -> None:
        """
        Main method to find, process, and generate embeddings for all slides in the slide_dir.
        """
        try:
            slide_filenames = self._get_slide_filenames()
            if not slide_filenames:
                logger.warning(f"No slides found in {self.slide_dir}. Aborting.")
                return

            output_path = self._create_output_directories(len(slide_filenames))
            segmenter, tile_embedder = self._initialize_processors(
                output_path, self.encoder
            )

            for slide_filename in tqdm(slide_filenames, desc="Processing slides"):
                self._process_single_slide(slide_filename, segmenter, tile_embedder)

            logger.info(colored("Processing complete.", "green"))
        except Exception as e:
            logger.error(
                colored(f"A critical error occurred: {e}", "red"), exc_info=True
            )

    def _get_slide_filenames(self) -> List[str]:
        """Retrieves a list of slide filenames from the source directory."""
        patient_folder = PatientFolder(str(self.slide_dir))
        return patient_folder.get_data()

    def _create_output_directories(self, num_slides: int) -> Path:
        """
        Creates the necessary output directories for the processing run.

        Args:
            num_slides (int): The number of slides being processed, used for naming.

        Returns:
            Path: The path to the main output directory for this run.
        """
        # IMPROVEMENT: Cleaner and more maintainable directory naming.
        run_name = (
            f"slides_{num_slides}_mag_{self.magnification}x_patch_{self.patch_size}"
        )
        output_path = self.output_dir / run_name

        # Create all subdirectories in one go
        for subdir in ["segmentation", "patches", "patch_embeddings"]:
            (output_path / subdir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Output will be saved to: {output_path}")
        return output_path

    def _initialize_processors(
        self, output_path: Path, encoder: str
    ) -> Tuple[TissueSegmenter, TileEmbedder]:
        """
        Initializes the tissue segmenter and tile embedder components.

        Args:
            output_path (Path): The base path for saving processor outputs.

        Returns:
            A tuple containing the initialized TissueSegmenter and TileEmbedder.
        """
        segmenter = TissueSegmenter(
            save_path=str(output_path / "segmentation"),
            batch_size=self.batch_size,
        )
        tile_embedder = TileEmbedder(
            model_name=encoder,
            target_patch_size=self.patch_size,
            target_mag=self.magnification,
            save_path=str(output_path),
        )
        return segmenter, tile_embedder

    def _process_single_slide(
        self,
        slide_filename: str,
        segmenter: TissueSegmenter,
        tile_embedder: TileEmbedder,
    ) -> None:
        """
        Processes a single WSI file: reads, segments, and embeds tiles.

        Args:
            slide_filename (str): The filename of the slide to process.
            segmenter (TissueSegmenter): The tissue segmentation processor.
            tile_embedder (TileEmbedder): The tile embedding processor.
        """
        try:
            logger.info(f"Processing slide: {slide_filename}")
            wsi = self._read_slide(slide_filename)
            pixel_size = get_pixel_size(wsi.img)
            # IMPROVEMENT: Use pathlib for cleaner file name manipulation.
            filename_no_extension = Path(slide_filename).stem

            gdf_contours = segmenter.segment_tissue(
                wsi=wsi,
                pixel_size=pixel_size,
                save_bn=filename_no_extension,
            )

            tile_embedder.embed_tiles(
                wsi=wsi,
                gdf_contours=gdf_contours,
                fn=filename_no_extension,
            )
        except openslide.OpenSlideError as e:
            logger.error(colored(f"Could not open slide {slide_filename}: {e}", "red"))
        except Exception as e:
            # IMPROVEMENT: exc_info=True provides a full traceback for better debugging.
            logger.error(
                colored(f"Error processing slide {slide_filename}: {e}", "red"),
                exc_info=True,
            )

    def _read_slide(self, slide_filename: str) -> OpenSlideWSI:
        """
        Opens a slide file using OpenSlide.

        Args:
            slide_filename (str): The filename of the slide.

        Returns:
            OpenSlideWSI: The wrapper object for the opened slide.
        """
        slide_path = self.slide_dir / slide_filename
        return OpenSlideWSI(openslide.OpenSlide(str(slide_path)))
