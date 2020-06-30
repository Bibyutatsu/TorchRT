"""Generate PGM image files for inference

This module is used to create PGM files from validation data of Fashion MNIST
dataset

    Usage:
        python generate_pgms.py -o images
"""

import argparse
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def main():
    """Creating PGM files for Inference

    This functions creates Inference Images from Fashion MNIST validation
    data
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-o", "--output", help="Path to the output directory.")
    parser.add_argument("-i", "--imfile", default="./", help="Image text file")
    parser.add_argument("-n", "--numimg", default=3000, help="Number of img")

    args, _ = parser.parse_known_args()
    output_dir = args.output
    img_file = args.imfile
    num_img = int(args.numimg)

    ds = torchvision.datasets.FashionMNIST(
        root=os.path.join(os.path.dirname(__file__), './Data/FashionMNIST'),
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    loader = DataLoader(ds, batch_size=1, num_workers=4, shuffle=False)
    tensor2pil = transforms.ToPILImage()

    with open(os.path.join(img_file, 'imagefile.txt'), 'w+') as imgfile:
        for i, (images, labels) in enumerate(loader):
            image = tensor2pil(images[0])
            label = labels.item()
            file_name = str(label) + "_" + str(i) + ".pgm"
            path = os.path.join(output_dir, file_name)
            image.save(path)

            imgfile.write(file_name + " " + str(label))

            if i >= num_img:
                break


if __name__ == '__main__':
    main()
