"""Data loading utilities for the OASIS MRI slice dataset.

This module exposes a ``torch.utils.data.Dataset`` implementation tuned for the
pre-processed PNG slices provided by the OASIS challenge as well as a
``make_loaders`` helper that constructs the three canonical train/validation
test data loaders with consistent transforms.
"""

import os, glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class OasisSlices(Dataset):
    """Dataset that reads grayscale PNG brain slices from a directory.

    The constructor discovers PNG files in the supplied directory, sorts them
    deterministically, and prepares a torchvision transform pipeline that
    normalises the pixel format expected by the VAE.
    """

    def __init__(self, root_dir, size=128):
        """Prepare the dataset.

        Args:
            root_dir: Directory containing PNG slices for one split.
            size: Target spatial resolution for each slice (square output).

        Raises:
            FileNotFoundError: If no PNG files are present in ``root_dir``.
        """
        self.paths = sorted(glob.glob(os.path.join(root_dir, "*.png")))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No PNGs in {root_dir}")
        self.tf = T.Compose([
            T.Grayscale(num_output_channels=1),
            # Resize every slice to a fixed square canvas using bilinear
            # interpolation so that the downstream network can rely on a
            # consistent spatial resolution.
            T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR),
            # Produce a float tensor scaled to [0, 1] that already matches the
            # single-channel expectation of the VAE encoder.
            T.ToTensor(),
        ])

    def __len__(self):
        """Return the total number of slices available in this split."""

        return len(self.paths)

    def __getitem__(self, i):
        """Load, transform, and return the ``i``-th slice as a tensor."""

        x = Image.open(self.paths[i])
        x = self.tf(x)  # Tensor shape: [1, H, W], dtype: float32, range: [0, 1]
        return x

def make_loaders(base_dir, size=128, bs=64, workers=1):
    """Create train/validation/test dataloaders with shared configuration.

    Args:
        base_dir: Root folder containing the three pre-defined dataset splits.
        size: Target image resolution fed to the VAE.
        bs: Batch size shared across the three data loaders.
        workers: Number of worker processes used by each ``DataLoader``.

    Returns:
        Tuple of ``DataLoader`` objects ``(train, val, test)``.
    """
    train = OasisSlices(os.path.join(base_dir, "keras_png_slices_train"), size)
    val   = OasisSlices(os.path.join(base_dir, "keras_png_slices_validate"), size)
    test  = OasisSlices(os.path.join(base_dir, "keras_png_slices_test"), size)
    dl = lambda ds, shuf: DataLoader(ds, batch_size=bs, shuffle=shuf,
                                     num_workers=workers, pin_memory=True)
    return dl(train, True), dl(val, False), dl(test, False)
