import pathlib
import torch
import torchvision

from . import common


_here = pathlib.Path(__file__).resolve().parent


# TODO
def _reshape(img):
    return img
    # # only works for imgs of size (3, 32, 32)
    # return img.view(3, 16, 2, 16, 2).transpose(1, 4).transpose(3, 4).reshape(12, 16, 16)


_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                             torchvision.transforms.Lambda(_reshape)])

_train_transform = torchvision.transforms.Compose([torchvision.transforms.ColorJitter(brightness=0.05, contrast=0.05,
                                                                                      saturation=0.05, hue=0.05),
                                                   torchvision.transforms.RandomRotation(10),
                                                   torchvision.transforms.RandomCrop(32, padding=4),
                                                   torchvision.transforms.RandomHorizontalFlip(),
                                                   _transform])


def cifar10(batch_size):
    train_dataset = torchvision.datasets.CIFAR10(_here / 'data/CIFAR10', download=True, train=True,
                                                 transform=_train_transform)
    test_dataset = torchvision.datasets.CIFAR10(_here / 'data/CIFAR10', download=True, train=False,
                                                transform=_transform)

    _test_len = len(test_dataset)
    test_len = int(0.5 * _test_len)
    val_len = _test_len - test_len
    # Results in an 80%/10%/10% split between train/val/test.
    test_dataset, val_dataset = torch.utils.data.random_split(test_dataset, (test_len, val_len))

    train_dataloader = common.dataloader(train_dataset, batch_size)
    val_dataloader = common.dataloader(val_dataset, batch_size)
    test_dataloader = common.dataloader(test_dataset, batch_size)

    img_size = (3, 32, 32)
    num_classes = 10

    return img_size, num_classes, train_dataloader, val_dataloader, test_dataloader
