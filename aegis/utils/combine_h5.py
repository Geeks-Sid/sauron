import h5py
import os
import glob
import argparse
from tqdm import tqdm

def combine_h5(source_dir, output_file):
    """
    Combines all .h5 files in source_dir into a single output_file.
    The structure of the output file will be:
    file_name (without extension) -> 'features' dataset
    """
    
    # Find all h5 files
    h5_files = glob.glob(os.path.join(source_dir, '*.h5'))
    
    if not h5_files:
        print(f"No .h5 files found in {source_dir}")
        return

    print(f"Found {len(h5_files)} files. Combining...")

    with h5py.File(output_file, 'w') as dest_h5:
        for file_path in tqdm(h5_files, desc="Processing files"):
            file_name = os.path.basename(file_path)
            base_name = os.path.splitext(file_name)[0]
            
            try:
                with h5py.File(file_path, 'r') as src_h5:
                    # Create a group for this file
                    grp = dest_h5.create_group(base_name)
                    
                    # Check if 'features' exists in source
                    # FAST
                    if 'features' in src_h5:
                        # This copies the dataset object directly, preserving attributes and 
                        # bypassing the read-to-RAM overhead.
                        src_h5.copy('features', grp, name='features')
                    else:
                        # If 'features' key not found, try to find any dataset
                        # This is a fallback mechanism
                        keys = list(src_h5.keys())
                        if len(keys) > 0:
                            # Use the first key found
                            first_key = keys[0]
                            src_h5.copy(first_key, grp, name='features')
                        else:
                            print(f"Warning: No datasets found in {file_name}")
                            
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    print(f"Successfully created {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple h5 files into one.")
    parser.add_argument('--source_dir', type=str, default='E:\\features_uni_v2', help="Directory containing source .h5 files")
    parser.add_argument('--output_file', type=str, default='E:\\combined_features.h5', help="Path to the output .h5 file")
    
    args = parser.parse_args()
    
    combine_h5(args.source_dir, args.output_file)
