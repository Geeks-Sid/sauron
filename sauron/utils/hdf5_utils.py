import pickle

import h5py
import numpy as np


def save_pickle(filename: str, obj: object) -> None:
    """
    Save an object to a pickle file.

    Args:
        filename (str): Filename for the pickle file.
        obj (object): Object to save.
    """
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


def save_hdf5(
    output_path, data_dict, attributes=None, mode="a", auto_chunk=True, chunk_size=None
):
    """
    Save data to an HDF5 file.

    Parameters:
    - output_path (str): Path to save the HDF5 file.
    - data_dict (dict): Dictionary of data to save.
    - attributes (dict, optional): Dictionary of attributes for each dataset.
    - mode (str): File open mode.
    - auto_chunk (bool): Use automatic chunking.
    - chunk_size (int, optional): Chunk size if auto_chunk is False.
    """
    with h5py.File(output_path, mode) as h5_file:
        for key, value in data_dict.items():
            value_shape = value.shape
            if value.ndim == 1:
                value = np.expand_dims(value, axis=1)
                value_shape = value.shape

            if key not in h5_file:
                data_type = (
                    h5py.string_dtype(encoding="utf-8")
                    if value.dtype == np.object_
                    else value.dtype
                )
                chunks = True if auto_chunk else (chunk_size,) + value_shape[1:]
                try:
                    dataset = h5_file.create_dataset(
                        key,
                        shape=value_shape,
                        chunks=chunks,
                        maxshape=(None,) + value_shape[1:],
                        dtype=data_type,
                    )
                    if attributes and key in attributes:
                        for attr_key, attr_value in attributes[key].items():
                            dataset.attrs[attr_key] = attr_value
                    dataset[:] = value
                except Exception as e:
                    print(f"Error encoding {key} of dtype {data_type} into HDF5: {e}")
            else:
                dataset = h5_file[key]
                dataset.resize(len(dataset) + value_shape[0], axis=0)
                assert dataset.dtype == value.dtype, "Data type mismatch"
                dataset[-value_shape[0] :] = value

    return output_path
