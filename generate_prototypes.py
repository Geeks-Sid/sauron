"""
This script performs K-means clustering on feature data extracted from whole-slide images.

It is optimized for memory efficiency during data loading and produces a structured
output file containing cluster centroids and metadata about the training run.

Key Features:
- Two clustering backends: scikit-learn's KMeans and the high-performance FAISS library.
- Memory-efficient data loading by pre-allocating arrays.
- Extracts and records the magnification level of patches used for clustering.
- Saves centroids and metadata (e.g., magnification counts, parameters) in a single,
  self-documenting PyTorch file.

Good reference for FAISS clustering:
https://github.com/facebookresearch/faiss/wiki/FAQ#questions-about-training
"""

import argparse
import collections
import glob
import logging
import os
import re
import time
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
from sklearn.cluster import KMeans

# --- Setup professional logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def load_data_optimized(
    root_pth: str, max_patches: int, mag_extract_pattern: str
) -> Tuple[np.ndarray, Dict]:
    """
    Loads H5 feature files efficiently and extracts metadata.

    Args:
        root_pth (str): Path to the root directory containing H5 files.
        max_patches (int): The maximum number of patches to load.
        mag_extract_pattern (str): Regex pattern to extract magnification from file path.
                                   Example: r'level_(\d+)'

    Returns:
        A tuple containing:
        - np.ndarray: A NumPy array of the loaded features.
        - dict: A dictionary containing metadata, like patch counts per magnification.
    """
    logging.info(f"Scanning for H5 files in: {root_pth}")
    pth_list = glob.glob(f"{root_pth}/**/*.h5", recursive=True)
    if not pth_list:
        raise FileNotFoundError(f"No .h5 files found in {root_pth}")

    # It's good practice to shuffle to avoid selection bias if data is ordered
    np.random.shuffle(pth_list)

    logging.info(f"Found {len(pth_list)} files. Loading up to {max_patches} patches.")

    # --- Pre-allocation Step ---
    # First, determine the feature dimension from the first valid file
    feature_dim = None
    for feat_path in pth_list:
        try:
            with h5py.File(feat_path, "r") as f:
                feature_dim = f["features"].shape[1]
                break
        except Exception as e:
            logging.warning(f"Could not read {feat_path} to determine feature dim: {e}")

    if feature_dim is None:
        raise IOError("Could not determine feature dimension from any H5 file.")

    # Pre-allocate the array for memory efficiency
    all_features = np.zeros((max_patches, feature_dim), dtype=np.float32)

    # --- Data Loading and Metadata Extraction Step ---
    magnification_counts = collections.Counter()
    num_loaded = 0

    for feat_path in pth_list:
        if num_loaded >= max_patches:
            break
        try:
            with h5py.File(feat_path, "r") as data_origin:
                features = np.asarray(data_origin["features"], dtype=np.float32)

                # Check for empty files
                if features.shape[0] == 0:
                    continue

                # Extract magnification
                match = re.search(mag_extract_pattern, feat_path)
                magnification = f"{match.group(1)}x" if match else "unknown"

                num_to_add = min(features.shape[0], max_patches - num_loaded)

                all_features[num_loaded : num_loaded + num_to_add] = features[
                    :num_to_add
                ]
                magnification_counts[magnification] += num_to_add
                num_loaded += num_to_add

        except Exception as e:
            logging.warning(f"Error loading {feat_path}: {e}")
            continue

    logging.info(f"Finished loading. Total patches loaded: {num_loaded}")

    # Trim the array to the actual number of loaded patches
    all_features = all_features[:num_loaded]

    metadata = {
        "magnification_counts": dict(magnification_counts),
        "total_patches_loaded": num_loaded,
        "feature_dimension": feature_dim,
    }

    return all_features, metadata


def main(args: argparse.Namespace):
    """Main execution function for clustering."""

    # Calculate total patches required and update data source path
    n_patches_target = args.n_proto * args.n_proto_patches
    data_source = args.data_source.replace("(project_name)", args.project)

    # --- Load Data ---
    patches_np, metadata = load_data_optimized(
        data_source, max_patches=n_patches_target, mag_extract_pattern=args.mag_pattern
    )
    logging.info(f"Shape of loaded data: {patches_np.shape}, Type: {patches_np.dtype}")
    logging.info(f"Magnification stats: {metadata['magnification_counts']}")

    if patches_np.shape[0] == 0:
        logging.error("No data was loaded. Exiting.")
        return

    # --- Clustering ---
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    centroids = None
    start_time = time.time()

    if args.mode == "kmeans":
        logging.info("Using scikit-learn KMeans for clustering...")
        logging.info(f"\tNum of clusters: {args.n_proto}, Max iter: {args.n_iter}")
        kmeans = KMeans(
            n_clusters=args.n_proto,
            max_iter=args.n_iter,
            n_init="auto",
            random_state=42,
        )
        kmeans.fit(patches_np)
        centroids = kmeans.cluster_centers_

    elif args.mode == "faiss":
        try:
            import faiss
        except ImportError:
            logging.error(
                "FAISS not installed. Please `pip install faiss-gpu` or `faiss-cpu`."
            )
            raise

        num_gpus = torch.cuda.device_count()
        logging.info(f"Using Faiss KMeans for clustering with {num_gpus} GPUs...")
        logging.info(
            f"\tNum of clusters: {args.n_proto}, Iterations: {args.n_iter}, Redos: {args.n_init}"
        )

        kmeans = faiss.Kmeans(
            d=patches_np.shape[1],
            k=args.n_proto,
            niter=args.n_iter,
            nredo=args.n_init,
            verbose=True,
            max_points_per_centroid=args.n_proto_patches,
            gpu=num_gpus,
            seed=42,
        )
        kmeans.train(patches_np)
        centroids = kmeans.centroids

    else:
        raise NotImplementedError(f"Clustering mode '{args.mode}' is not implemented.")

    end_time = time.time()
    logging.info(f"Clustering took {end_time - start_time:.2f} seconds.")
    logging.info(f"Shape of calculated centroids (prototypes): {centroids.shape}")

    # --- Save Prototypes and Metadata ---
    # Convert final centroids to a torch tensor
    centroids_tensor = torch.from_numpy(centroids)

    # Prepare the structured dictionary for saving
    output_data = {
        "centroids": centroids_tensor,
        "metadata": {
            **metadata,  # Include metadata from data loading
            "clustering_mode": args.mode,
            "project": args.project,
            "n_proto": args.n_proto,
            "n_proto_patches": args.n_proto_patches,
            "n_iter": args.n_iter,
            "n_init": args.n_init,
            "data_source": data_source,
            "clustering_time_seconds": end_time - start_time,
        },
    }

    save_fpath = args.save_path.replace("(project_name)", args.project)
    torch.save(output_data, save_fpath)
    logging.info(f"Successfully saved prototypes and metadata to: {save_fpath}")


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Configurations for K-Means Prototype Generation"
    )

    # --- Key Arguments ---
    parser.add_argument(
        "--project",
        type=str,
        default="PAAD",
        help="Project name, used for sourcing data and naming output.",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="path/to/data_root_dir/features_TCGA_256/feature_(project_name)_256_uni_dim_1024",
        help="Path pattern to the data source directory. Use '(project_name)' as a placeholder.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./prototypes/(project_name)_prototypes.pt",
        help="Path to save the final prototypes file. Use '(project_name)' as a placeholder.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["kmeans", "faiss"],
        default="kmeans",
        help="Clustering backend to use.",
    )

    # --- Clustering Parameters ---
    parser.add_argument(
        "--n_proto",
        type=int,
        default=16,
        help="Number of prototypes (clusters) to generate.",
    )
    parser.add_argument(
        "--n_proto_patches",
        type=int,
        default=10000,
        help="Number of patches per prototype to aim for. Total patches = n_proto * n_proto_patches.",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=50,
        help="Number of iterations for K-means clustering.",
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=5,
        help="Number of different initializations (for FAISS nredo). scikit-learn's `n_init` is now handled automatically.",
    )

    # --- New Metadata Argument ---
    parser.add_argument(
        "--mag_pattern",
        type=str,
        default=r"_(\d+)x",
        help="Regex pattern to extract magnification from file paths (e.g., 'TCGA-2A-A8VL-01Z-00-DX1_10x_...'). The first captured group is used.",
    )

    args = parser.parse_args()

    main(args)
