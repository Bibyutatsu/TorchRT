"""Utility module for various dataset related functions

Module containing dataset related functions to get Fashion-MNIST dataset
"""

import torchvision
import torchvision.transforms as transforms
import os


def get_data(train=False):
    """Loads the Fashion MNIST dataset

    Args:
        train: Optional, bool to load the training dataset
    Returns:
        ds: Fashion-MNIST dataset
    """

    ds = torchvision.datasets.FashionMNIST(
        root=os.path.join(os.path.dirname(__file__), './Data/FashionMNIST'),
        train=train,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    return ds


def get_classes():
    """Get the class labels in Fashion-MNIST dataset

    Returns:
        classes: The class labels of Fashion-MNIST
    """
    val = get_data()
    return val.classes
