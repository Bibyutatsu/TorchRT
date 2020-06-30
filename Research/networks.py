""" The module containing all the PyTorch Models

I used three different models for this project and can be listed here as:
1. torNet
2. FashionCNN
3. EfficientNet

    Typical usage:

    from networks import foo
"""

import torch.nn as nn
import torch.nn.functional as F
import torchvision
from efficientnet_pytorch import EfficientNet


class torNet(nn.Module):
    """torNet model

    A resnet50 model
    """

    def __init__(self, num_channels=1, num_classes=10):
        super(torNet, self).__init__()
        self.model = torchvision.models.resnet50()

        # Change the initial conv1 layer to have 1 layer instead of 3
        self.model.conv1 = nn.Conv2d(num_channels,
                                     64,
                                     kernel_size=(7, 7),
                                     stride=(2, 2),
                                     padding=(3, 3),
                                     bias=False)
        self.out = nn.Linear(in_features=1000, out_features=num_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.relu(x)
        x = self.out(x)
        return x


class efficientNet(nn.Module):
    """ EfficientNet model with b=3

    EfficientNet model with 12 Million parameters
    """

    def __init__(self, num_channels=1, num_classes=10, image_size=(28, 28)):
        super(efficientNet, self).__init__()
        self.model = EfficientNet.from_name("efficientnet-b3", in_channels=1)
        self.out = nn.Linear(in_features=1000, out_features=num_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.relu(x)
        x = self.out(x)
        return x
    
    def onnx_swish(self):
        self.model.set_swish(memory_efficient=False)


class FashionCNN(nn.Module):
    """FashionCNN model

    A simple CNN model which I have taken from:
    https://www.kaggle.com/pankajj/fashion-mnist-with-pytorch-93-accuracy
    """

    def __init__(self):
        super(FashionCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.flatten(1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out
