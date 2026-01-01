import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tinyimagenet import TinyImageNet
from torch.utils.data.distributed import DistributedSampler

cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std  = (0.2470, 0.2435, 0.2616)

cifar_10_train_transformations = v2.Compose([
    v2.RandomHorizontalFlip(p=0.2),
    v2.RandomApply([v2.RandomAffine(degrees=15,translate=(0.1,0.1),scale=(0.75,1.25),shear=10)],p=0.2),
    v2.ToImage(),
    v2.ToDtype(torch.float32,scale=True),
    v2.Normalize(cifar_mean,cifar_std)])

cifar_10_test_transformations = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32,scale=True),
    v2.Normalize(cifar_mean,cifar_std)])


def cifar_10_dataloader(batch_size=16,n_workers=0):
    train_dataset = torchvision.datasets.CIFAR10(root = './cifar10_data',train=True, transform=cifar_10_train_transformations, target_transform=None, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root = './cifar10_data',train=False, transform=cifar_10_test_transformations, target_transform=None, download=True)

    print(f" \nthe class are   {'--'.join(train_dataset.classes)}\n")

    print(f"{train_dataset.data.shape=}")
    print(f"{test_dataset.data.shape=}")
    train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False,num_workers=n_workers)
    test_loader = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)

    return train_dataset, test_dataset, train_loader, test_loader



tiny_imagenent_mean = (0.480, 0.448, 0.398)
tiny_imagenent_std  = (0.276, 0.269, 0.282)
tiny_imagenet_train_transformations = v2.Compose([
    v2.RandomHorizontalFlip(p=0.2),
    v2.RandomApply([v2.RandomAffine(degrees=15,translate=(0.1,0.1),scale=(0.75,1.25),shear=10)],p=0.2),
    v2.ToImage(),
    v2.ToDtype(torch.float32,scale=True),
    v2.Normalize(cifar_mean,cifar_std)])

tiny_imagenet_test_transformations = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32,scale=True),
    v2.Normalize(tiny_imagenent_mean,tiny_imagenent_std)])

def tiny_imagenet_dataloader(batch_size=16,distributed=False,n_workers=0):
    train_dataset = TinyImageNet(root='./tiny_imagenet_data',split='train',transform=tiny_imagenet_train_transformations)
    test_dataset = TinyImageNet(root='./tiny_imagenet_data',split='val',transform=tiny_imagenet_test_transformations)

    data_sampler=None
    if distributed:
        data_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=False if data_sampler else True, pin_memory=False,num_workers=n_workers,sampler=data_sampler,drop_last=True)
    test_loader = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)

    return train_dataset, test_dataset, train_loader, test_loader, data_sampler



import matplotlib.pyplot as plt
def view_sample_images(train_dataset):
    nrow, ncol = 4,8
    fig,axes = plt.subplots(nrows=nrow,ncols=ncol,sharex=True,sharey=True)
    for i in range(nrow):
        for j in range(ncol):
            axes[i][j].imshow(train_dataset.data[i*ncol+j])


