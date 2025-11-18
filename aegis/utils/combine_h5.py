import argparse
import glob
import os

import h5py
from tqdm import tqdm


def copy_h5_object(src_obj, dest_group, name):
    """
    Recursively copy an HDF5 object (dataset or group) from source to destination.
    Handles nested groups properly.
    """
    if isinstance(src_obj, h5py.Dataset):
        # Copy dataset using h5py's copy method
        try:
            dest_group.copy(src_obj, name)
        except Exception:
            # Fallback: read data and create new dataset
            data = src_obj[...]
            dset = dest_group.create_dataset(
                name, data=data, compression=src_obj.compression
            )
            # Copy attributes
            for attr_name in src_obj.attrs.keys():
                dset.attrs[attr_name] = src_obj.attrs[attr_name]
    elif isinstance(src_obj, h5py.Group):
        # Create a new group and recursively copy its contents
        new_group = dest_group.create_group(name)
        for key in src_obj.keys():
            copy_h5_object(src_obj[key], new_group, key)
        # Copy attributes from the source group
        for attr_name in src_obj.attrs.keys():
            new_group.attrs[attr_name] = src_obj.attrs[attr_name]
    else:
        raise ValueError(f"Unknown HDF5 object type: {type(src_obj)}")


def combine_h5(source_dir, output_file):
    """
    Combines all .h5 files in source_dir into a single output_file.
    The structure of the output file will be:
    file_name (without extension) -> all datasets and groups from source file
    """

    # Find all h5 files
    h5_files = glob.glob(os.path.join(source_dir, "*.h5"))

    if not h5_files:
        print(f"No .h5 files found in {source_dir}")
        return

    print(f"Found {len(h5_files)} files. Combining...")

    with h5py.File(output_file, "w") as dest_h5:
        for file_path in tqdm(h5_files, desc="Processing files"):
            file_name = os.path.basename(file_path)
            base_name = os.path.splitext(file_name)[0]

            try:
                # First, try to open the file to check if it's valid
                with h5py.File(file_path, "r") as src_h5:
                    # Create a group for this file
                    grp = dest_h5.create_group(base_name)

                    # Copy all objects (datasets and groups) from source file
                    # Use recursive copy to handle nested structures properly
                    for key in src_h5.keys():
                        try:
                            copy_h5_object(src_h5[key], grp, key)
                        except Exception as copy_error:
                            print(
                                f"  Warning: Failed to copy '{key}' from {file_name}: {copy_error}"
                            )
                            # Continue with other keys even if one fails

            except OSError as e:
                print(f"Error: Cannot open {file_name} - file may be corrupted: {e}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    print(f"Successfully created {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple h5 files into one.")
    parser.add_argument(
        "--source_dir",
        type=str,
        default="E:\\features_uni_v2",
        help="Directory containing source .h5 files",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="E:\\combined_features.h5",
        help="Path to the output .h5 file",
    )

    args = parser.parse_args()

    combine_h5(args.source_dir, args.output_file)
