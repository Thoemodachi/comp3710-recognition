import os, glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class OasisSlices(Dataset):
    def __init__(self, root_dir, size=128):
        self.paths = sorted(glob.glob(os.path.join(root_dir, "*.png")))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No PNGs in {root_dir}")
        self.tf = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),                       # [0,1]
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        x = Image.open(self.paths[i])
        x = self.tf(x)                         # [1,H,W], float32 in [0,1]
        return x

def make_loaders(base_dir, size=128, bs=64, workers=1):
    train = OasisSlices(os.path.join(base_dir, "keras_png_slices_train"), size)
    val   = OasisSlices(os.path.join(base_dir, "keras_png_slices_validate"), size)
    test  = OasisSlices(os.path.join(base_dir, "keras_png_slices_test"), size)
    dl = lambda ds, shuf: DataLoader(ds, batch_size=bs, shuffle=shuf,
                                     num_workers=workers, pin_memory=True)
    return dl(train, True), dl(val, False), dl(test, False)
