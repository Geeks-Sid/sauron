import h5py
from PIL import Image
from torch.utils.data import Dataset

from sauron.utils.WSIObjects import OpenSlideWSIPatcher, get_pixel_size


class TileDataset(Dataset):
    def __init__(
        self,
        wsi,
        contours,
        target_patch_size,
        target_magnification,
        eval_transform,
        save_path=None,
    ):
        self.wsi = wsi
        self.contours = contours
        self.eval_transform = eval_transform

        self.patcher = OpenSlideWSIPatcher(
            wsi=wsi,
            patch_size=target_patch_size,
            src_pixel_size=get_pixel_size(wsi.img),
            dst_pixel_size=self.magnification_to_pixel_size(target_magnification),
            mask=contours,
            coords_only=False,
        )
        self.patcher.save_visualization(path=save_path)

    @staticmethod
    def magnification_to_pixel_size(magnification):
        magnification_map = {2.5: 4.0, 5: 2.0, 10: 1.0, 20: 0.5, 40: 0.25}
        if magnification in magnification_map:
            return magnification_map[magnification]
        else:
            raise ValueError("Magnification should be in [2.5, 5, 10, 20, 40].")

    def _load_coords(self):
        with h5py.File(self.coords_h5_fpath, "r") as f:
            self.attr_dict = {
                k: dict(f[k].attrs) for k in f.keys() if len(f[k].attrs) > 0
            }
            self.coords = f["coords"][:]
            self.patch_size = f["coords"].attrs["patch_size"]
            self.custom_downsample = f["coords"].attrs["custom_downsample"]
            self.target_patch_size = (
                int(self.patch_size) // int(self.custom_downsample)
                if self.custom_downsample > 1
                else self.patch_size
            )

    def __len__(self):
        return len(self.patcher)

    def __getitem__(self, index):
        image, x, y = self.patcher[index]
        image = Image.fromarray(image, "RGB")
        image = self.eval_transform(image).unsqueeze(dim=0)
        return image, (x, y)
