import GPUtil as GPUtil
import numpy as np
import torch as torch
from networks import Discriminator
import torch.nn as nn
import random
from dataset import Data
from torch.utils.data import DataLoader
import os

def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=50, lambda_relative=1.0):
    """
    Trains the regression based Motion Artifact Quantification model.
    :param model: the model to be trained;
    :param train_loader: train dataloader;
    :param val_loader: validation dataloader;
    :param criterion: loss criterion;
    :param optimizer: optimizer;
    :param lambda_relative: relative weight of the ranking loss in the total loss;
    :return: train and validation loss history
    """
    loss_history = []
    val_losses = []

    for epoch in range(num_epochs):
        print("Epoch [%d] start" % (epoch))
        GPUtil.showUtilization()
        model.train()
        losses = []
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for i_step, (data, target) in enumerate(train_loader):
            print(str(i_step) + '...')
            data_base = data[0].half().to(device)
            data_bad = data[1].half().to(device)
            target_base = target[0].to(device)
            target_bad = target[1].to(device)

            output_base, output_bad = model(data_base), model(data_bad)

            target_base = target_base.type_as(output_base)
            target_bad = target_bad.type_as(output_bad)

            loss = criterion(output_base, target_base.unsqueeze(1))
            loss += criterion(output_bad, target_bad.unsqueeze(1))
            z = torch.tensor(0.0).cuda().half()

            loss += torch.mean(torch.where((output_base - output_bad)>0.0,
                                           (output_base - output_bad), z))*lambda_relative
            if torch.mean(output_base-output_bad)>0:
                loss += torch.mean(output_base-output_bad)*lambda_relative

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = _compute_val(model, val_loader)
        val_losses.append(val_loss)
        loss_history.append(np.array(losses).mean())

        print("Epoch [%d]" % (epoch))
        print("Mean loss on train:", np.array(losses).mean(),
              "Mean loss on val:", val_loss,
              )

    return loss_history, val_losses


def _compute_val(model, loader, device, lambda_relative = 1.0):
    """
    Computes validation loss for the Motion Artifact Quantification Model
    :param model: model;
    :param loader: data loader;
    :param lambda_relative: ranking loss relative weight in the overall validation loss
    :return: validation loss
    """
    val_loss = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for i_step, (data, target) in enumerate(loader):
            data_base = data[0].to(device).half()
            data_bad = data[1].to(device).half()
            target_base = target[0].to(device)
            target_bad = target[1].to(device)

            output_base, output_bad = model(data_base.half()), model(data_bad.half())
            target_base = target_base.type_as(output_base)
            target_bad = target_bad.type_as(output_bad)

            criterion = nn.MSELoss()
            val_loss += criterion(output_base, target_base.unsqueeze(1))
            val_loss += criterion(output_bad, target_bad.unsqueeze(1))

            z = torch.tensor(0.0).cuda().half()
            val_loss += torch.mean(torch.where((output_base - output_bad)>0.0,
                                               (output_base - output_bad), z))*lambda_relative

            if torch.mean(output_base-output_bad)>0:
                val_loss += torch.mean(output_base-output_bad)*lambda_relative

    return (val_loss) / (i_step + 1)


def discriminator(degrees1, degrees2, degrees3,
                  translation1, translation2, translation3,train_set,
                  num_transforms=1):
    """
    Calculates the discriminator loss for the CNN network differentiating between the real bad
    and simulated bad 3D images.
    :param degrees1: 3D motion rotation axis x;
    :param degrees2: 3D motion rotation axis y;
    :param degrees3: 3D motion rotation axis z;
    :param translation1: 3D motion translation axis x;
    :param translation2: 3D motion translation axis y;
    :param translation3: 3D motion translation axis z;
    :param train_set: list of train set image names;
    :param num_transforms: 3D motion, number of moves;
    :return: discriminator loss when differentiating between real bad and simulated bad images
    """
    train_dataset = Data(inputs=train_set, rotate=(degrees1, degrees2, degrees3),
                         translate=(translation1, translation2, translation3), number=1)

    train_dataloader = DataLoader(train_dataset, batch_size=6, num_workers=12, shuffle=True)
    loss_history = []
    model = Discriminator()
    p_net = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        p_net.cuda()

    optimizer = torch.optim.Adam(p_net.parameters(), lr=1e-4, eps=1e-4)
    p_net = p_net.half()

    # Training:
    for epoch in range(10):
        print("Epoch [%d] start" % (epoch))

        p_net.train()
        losses = []
        for i_step, (data, target) in enumerate(train_dataloader):
            outputs1 = p_net(data[0]).half()
            outputs2 = p_net(data[1]).half()
            target1 = target[0].to(device).reshape((outputs1.shape)).half()
            target2 = target[1].to(device).reshape((outputs2.shape)).half()
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(outputs1, target1)
            loss += criterion(outputs2, target2)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_history.append(np.array(losses).mean())
    print('Finished Training')
    print(loss_history)
    return np.array(losses).mean()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def normalize(img):
    '''Minmax image normalization'''
    return (img- torch.min(img))/(torch.max(img)-torch.min(img))


def efc(img, framemask=None):
    """
    Copied from here:

    https://mriqc.readthedocs.io/en/latest/_modules/mriqc/qc/anatomical.html#efc
    Calculate the :abbr:`EFC (Entropy Focus Criterion)` [Atkinson1997]_.
    Uses the Shannon entropy of voxel intensities as an indication of ghosting
    and blurring induced by head motion. A range of low values is better,
    with EFC = 0 for all the energy concentrated in one pixel.

    .. math::

        \text{E} = - \sum_{j=1}^N \frac{x_j}{x_\text{max}}
        \ln \left[\frac{x_j}{x_\text{max}}\right]

    with :math:`x_\text{max} = \sqrt{\sum_{j=1}^N x^2_j}`.

    The original equation is normalized by the maximum entropy, so that the
    :abbr:`EFC (Entropy Focus Criterion)` can be compared across images with
    different dimensions:

    .. math::

        \text{EFC} = \left( \frac{N}{\sqrt{N}} \, \log{\sqrt{N}^{-1}} \right) \text{E}

    :param numpy.ndarray img: input data
    :param numpy.ndarray framemask: a mask of empty voxels inserted after a rotation of
      data

    """
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    if framemask is None:
        framemask = np.zeros_like(img, dtype=np.uint8)

    n_vox = np.sum(1 - framemask)
    # print(np.log(1.0 / np.sqrt(n_vox)))
    # Calculate the maximum value of the EFC (which occurs any time all
    # voxels have the same value)
    efc_max = 1.0 * n_vox * (1.0 / np.sqrt(n_vox)) * \
              np.log(1.0 / np.sqrt(n_vox))
    # Calculate the total image energy
    b_max = np.sqrt((img[framemask == 0] ** 2).sum())

    # Calculate EFC (add 1e-16 to the image data to keep log happy)
    return float((1.0 / efc_max) * np.sum((img[framemask == 0] / b_max) * np.log(
        (img[framemask == 0] + 1e-16) / b_max)))


def efc_new(img, framemask=None):
    """
    efc modified to be looking only at the non black part;
    """
    quant98 = quantile(img, 0.98)
    quant30 = quantile(img, 0.3)
    img[img > quant98] = quant98
    img = img - quant30
    img = img[img > 0]
    if framemask is None:
        framemask = np.zeros_like(img, dtype=np.uint8)
    n_vox = np.sum(1 - framemask)
    # Calculate the maximum value of the EFC (which occurs any time all
    # voxels have the same value)
    efc_max = 1.0 * n_vox * (1.0 / np.sqrt(n_vox)) * \
              np.log(1.0 / np.sqrt(n_vox))
    # Calculate the total image energy
    b_max = np.sqrt((img[framemask == 0] ** 2).sum())

    # Calculate EFC (add 1e-16 to the image data to keep log happy)
    return float((1.0 / efc_max) * np.sum((img[framemask == 0] / b_max) * np.log(
        (img[framemask == 0] + 1e-16) / b_max)))

def quantile(img, x):
    img = img.flatten()
    img.sort()
    q = x * (len(img) + 1)
    q_int = int(q)
    res = (1 - (q - q_int)) * img[q_int - 1] + (q - q_int) * img[q_int]
    return res
