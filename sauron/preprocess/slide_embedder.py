import logging
import os

import openslide
from termcolor import colored
from tqdm import tqdm

from sauron.preprocess.compute_embeddings import TileEmbedder
from sauron.utils.filehandler import PatientFolder
from sauron.utils.segmentation import TissueSegmenter
from sauron.utils.WSIObjects import OpenSlideWSI, get_pixel_size

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SlideEmbedder:
    def __init__(self, slide_dir, output_dir, magnification, patch_size, batch_size):
        self.slide_dir = slide_dir
        self.output_dir = output_dir
        self.magnification = magnification
        self.patch_size = patch_size
        self.batch_size = batch_size

    def process_slides(self):
        try:
            slide_filenames = self._get_slide_filenames()
            output_path = self._create_output_directories(slide_filenames)
            segmenter, tile_embedder = self._initialize_processors(output_path)

            for slide_filename in tqdm(slide_filenames, desc="Processing slides"):
                self._process_single_slide(slide_filename, segmenter, tile_embedder)

            logger.info(colored("Processing complete.", "green"))
        except Exception as e:
            logger.error(colored(f"Failed to process slides: {e}", "red"))

    def _get_slide_filenames(self):
        patient_folder = PatientFolder(self.slide_dir)
        return patient_folder.get_data()

    def _create_output_directories(self, slide_filenames):
        output_path = os.path.join(
            self.output_dir,
            f"processing_nWSI_{len(slide_filenames)}_mag_{self.magnification}x_patchsize_{self.patch_size}",
        )
        os.makedirs(os.path.join(output_path, "segmentation"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "patches"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "patch_embeddings"), exist_ok=True)
        return output_path

    def _initialize_processors(self, output_path):
        segmenter = TissueSegmenter(
            save_path=os.path.join(output_path, "segmentation"),
            batch_size=self.batch_size,
        )
        tile_embedder = TileEmbedder(
            target_patch_size=self.patch_size,
            target_mag=self.magnification,
            save_path=output_path,
        )
        return segmenter, tile_embedder

    def _process_single_slide(self, slide_filename, segmenter, tile_embedder):
        try:
            wsi = self._read_slide(slide_filename)
            pixel_size = get_pixel_size(wsi.img)
            filename_no_extension = os.path.splitext(slide_filename)[0]

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
        except Exception as e:
            logger.error(
                colored(f"Error processing slide {slide_filename}: {e}", "red")
            )

    def _read_slide(self, slide_filename):
        slide_path = os.path.join(self.slide_dir, slide_filename)
        return OpenSlideWSI(openslide.OpenSlide(slide_path))
