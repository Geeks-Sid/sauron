import argparse
import os

import h5py


def print_dataset_shapes(obj, path="/", indent=0):
    """
    Recursively traverse HDF5 file structure and print all dataset shapes.

    Args:
        obj: HDF5 object (File, Group, or Dataset)
        path: Current path in the HDF5 hierarchy
        indent: Indentation level for pretty printing
    """
    indent_str = "  " * indent

    if isinstance(obj, h5py.Dataset):
        # Print dataset information
        shape = obj.shape
        dtype = obj.dtype
        print(f"{indent_str}{path}: shape={shape}, dtype={dtype}")
    elif isinstance(obj, h5py.Group):
        # Recursively process groups
        for key in obj.keys():
            new_path = f"{path}/{key}" if path != "/" else f"/{key}"
            print_dataset_shapes(obj[key], new_path, indent + 1)


def inspect_h5(h5_file):
    """
    Inspect a combined H5 file and print all dataset shapes.

    Args:
        h5_file: Path to the H5 file to inspect
    """
    if not os.path.exists(h5_file):
        print(f"Error: File not found: {h5_file}")
        return

    print(f"Inspecting: {h5_file}\n")
    print("=" * 80)

    try:
        with h5py.File(h5_file, "r") as f:
            # Count total datasets
            dataset_count = 0

            def count_datasets(name, obj):
                nonlocal dataset_count
                if isinstance(obj, h5py.Dataset):
                    dataset_count += 1

            f.visititems(count_datasets)

            print(f"Total datasets found: {dataset_count}\n")
            print("Dataset shapes:")
            print("=" * 80)

            # Print all dataset shapes
            print_dataset_shapes(f, "/", indent=0)

            print("=" * 80)
            print(
                f"\nSummary: {dataset_count} datasets found in {len(f.keys())} top-level groups"
            )

    except OSError as e:
        print(f"Error: Cannot open file - file may be corrupted: {e}")
    except Exception as e:
        print(f"Error inspecting file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect combined H5 file and print all dataset shapes."
    )
    parser.add_argument(
        "--h5_file",
        type=str,
        default="E:\\combined_features.h5",
        help="Path to the combined H5 file to inspect",
    )

    args = parser.parse_args()

    inspect_h5(args.h5_file)
