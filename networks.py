import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    Network used for the Motion Artifact Severity Prediction
    """
    def __init__(self):
        # 128x128x128
        super(Net, self).__init__()
        self.conv_down1 = double_conv(1, 32)
        self.conv_down2 = double_conv(32, 64)
        self.conv_down3 = double_conv(64, 128)
        self.conv_down4 = double_conv(128, 256)

        self.maxpool = nn.MaxPool3d((2, 2, 2))
        self.fc1 = nn.Linear(16 * 16 * 16 * 256, 1)
        self.m = nn.Sigmoid()

    def forward(self, x):
        inp = x
        x = x.half()
        conv1 = self.conv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.conv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.conv_down3(x)
        x = self.maxpool(conv3)
        x = self.conv_down4(x)

        x = x.view(-1, 16 * 16 * 16 * 256)
        x = self.fc1(x)

        return x
def double_conv(in_channels: int, out_channels: int) -> object:
    '''
    Helper function for the 3D Net
    '''
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True))

class Discriminator(nn.Module):
    """
    Network used for discriminating between the real bad and the simulated bad images
    """
    def __init__(self):
        #Input shape: 128x128x128
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(1, 5, 3, padding = 1)
        self.pool = nn.MaxPool3d((2, 2,2))
        self.conv2 = nn.Conv3d(5, 10, 3, padding = 1)
        self.conv3 = nn.Conv3d(10, 20, 3, padding = 1)
        self.fc1 = nn.Linear(16*16*16*20, 1)

    def forward(self, x):
        x = x.half()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 16*16*16*20)
        x = (self.fc1(x))
        return x