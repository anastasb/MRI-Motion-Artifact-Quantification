import torch
import numpy as np
from torch.utils.data import Dataset
import torchio as tio
from utils import normalize
import albumentations as alb
import nibabel as nib

"""
PyTorch Dataset generator class to be used in DataLoader
"""

SIZE = 128
DEPTH = 128
BASE_DIR = 'usable_t1_baseline/'
class Data(Dataset):
    """
    inputs: list of filenames to each image;
    type:  train/validation dataset type;
    normalize: if True, normalize
    """
    torch.manual_seed(2018)
    # Constructor
    def __init__(self, inputs: list,
                 normalize=True, data_type='train'):
        self.inputs = inputs
        self.type = data_type
        self.normalize = normalize

    # Getter
    def __getitem__(self, index):
        torch.manual_seed(2018)
        X_img1 = nib.load(BASE_DIR + self.inputs[index])
        y1 = torch.tensor(0.0, dtype=torch.float)
        parameter = 0.5 * torch.rand(1) #severity of the transform
        transform = tio.Motion(degrees=[np.array((parameter, parameter, parameter))],
                               translation=[np.array((parameter, parameter, parameter))],
                               times=torch.arange(0, 1, 0.5)[1:].numpy(), image_interpolation='linear')

        image = tio.Image(BASE_DIR + self.inputs[index])
        transformed_img = transform(image)

        X_array = X_img1.get_fdata() if len(X_img1.get_fdata().shape) < 4 else X_img1.get_fdata().squeeze(-1)
        X_array2 = transformed_img.data if len(transformed_img.data.shape) < 4 else transformed_img.data.squeeze(-1)

        if len(X_array.shape) > 3:
            X_array = torch.tensor(X_array).squeeze(0)

        if len(X_array2.shape) > 3:
            X_array2 = torch.tensor(X_array2).squeeze(0)

        X_depth = X_img1.shape[2]
        pad = (X_depth - DEPTH) // 2
        idx = range(pad, pad + DEPTH)
        resize = alb.Resize(width=SIZE, height=SIZE, p=1.0)
        X_array = resize(image=np.array(X_array[:, :, idx]))['image']
        X_array2 = resize(image=np.array(X_array2[:, :, idx]))['image']

        augment = torch.rand(2)
        hflip = alb.HorizontalFlip(p=1)
        vflip = alb.VerticalFlip(p=1)
        if self.type == 'train' and augment[0] > 0.5:
            X_array = hflip(image=X_array)['image']
            X_array2 = hflip(image=X_array2)['image']

        if self.type == 'train' and augment[1] > 0.5:
            X_array = vflip(image=X_array)['image']
            X_array2 = vflip(image=X_array2)['image']

        X_array = torch.tensor(X_array)
        X_array2 = torch.tensor(X_array2)

        if self.normalize:  # normalize during both training and testing
            X_array = normalize(X_array)
            X_array2 = normalize(X_array2)

        y2 = torch.tensor(abs(parameter), dtype=torch.float)
        return (X_array.unsqueeze(0).float(), X_array2.unsqueeze(0).float()), (y1, y2)

    # Get length
    def __len__(self):
        return len(self.inputs)