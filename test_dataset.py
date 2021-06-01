import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchio as tio
import albumentations as alb
import random

"""
Synthetic Test Dataset generator class
"""

SIZE = 128
DEPTH = 128
BASE_DIR = 'usable_t1_baseline/'
class TestData(Dataset):
    """
    inputs: list of filenames to each image;
    normalize: if True, normalize
    """
    # Constructor
    def __init__(self, inputs: list,
                 normalize=True):
        self.inputs = inputs
        self.normalize = normalize

    # Getter
    def __getitem__(self, index):
        parameter = torch.rand(1)  # transform parameter, includes extrapolation
        transform = tio.Motion(degrees=[np.array((parameter, parameter, parameter))],
                               translation=[np.array((parameter, parameter, parameter))],
                               times=torch.arange(0, 1, 0.5)[1:].numpy(), image_interpolation='linear')
        image = tio.Image(BASE_DIR + self.inputs[index])
        transformed_img = transform(image)
        X_array = transformed_img.data if len(transformed_img.data.shape) < 4 else transformed_img.data.squeeze(-1)
        if len(X_array.shape) > 3:
            X_array = torch.tensor(X_array).squeeze(0)

        X_depth = X_array.shape[2]
        pad = (X_depth - DEPTH) // 2
        idx = range(pad, pad + DEPTH)
        X_array = resize(image=np.array(X_array[:, :, idx]))['image']
        X_array = torch.tensor(X_array)
        if self.normalize:
            X_array = normalize(X_array)
        y = torch.tensor(parameter, dtype=torch.float)
        return X_array.unsqueeze(0).float(), y

    # Get length
    def __len__(self):
        return len(self.inputs)