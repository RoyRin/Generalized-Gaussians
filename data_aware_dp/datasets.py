import numpy as np
import torch
from torch.utils.data import Subset
import torchvision
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

BATCH_SIZE = 128


def get_LR_train_and_test_data(n_samples=10000,
                               input_dim=1,
                               train_percentage=0.8,
                               noise=0,
                               test_noise=0):
    train_samples = int(n_samples * train_percentage)
    test_samples = int(n_samples * train_percentage)

    X_train, Y_train = make_regression(n_samples=train_samples,
                                       n_features=input_dim,
                                       n_informative=input_dim,
                                       bias=0,
                                       noise=noise)
    X_test, Y_test = make_regression(n_samples=test_samples,
                                     n_features=input_dim,
                                     n_informative=input_dim,
                                     bias=0,
                                     noise=test_noise)

    #X_train = X[:int(n_samples * train_percentage)]
    #Y_train = Y[:int(n_samples * train_percentage)]
    #X_test = X[int(n_samples * train_percentage):]
    #Y_test = Y[int(n_samples * train_percentage):]

    return (X_train, Y_train), (X_test, Y_test)


def plot_LR_test_and_train(model, X_test, Y_test, X_train, Y_train):
    plt.title("train")
    plt.scatter(X_train[:, 0], Y_train)
    xrange = torch.linspace(-4, 4, 100)
    plt.plot(xrange.numpy(),
             model(xrange.unsqueeze(1).double()).detach().numpy(),
             c='r')
    plt.show()

    plt.title("test")
    plt.scatter(X_test[:, 0], Y_test)
    xrange = torch.linspace(-4, 4, 100)
    plt.plot(xrange.numpy(),
             model(xrange.unsqueeze(1).double()).detach().numpy(),
             c='r')
    plt.show()


def get_dataloader(X, Y, batch_size=128):
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.from_numpy(X), torch.from_numpy(Y)),
                                       batch_size=batch_size,
                                       shuffle=True)


def get_LR_dataloader(n_samples=1000,
                      input_dim=10,
                      noise=30,
                      batch_size=BATCH_SIZE):

    (X_train,
     Y_train), (X_test,
                Y_test) = get_LR_train_and_test_data(n_samples=n_samples,
                                                     input_dim=input_dim,
                                                     train_percentage=0.8,
                                                     noise=noise,
                                                     test_noise=0)

    trainloader = get_dataloader(X_train, Y_train, batch_size=batch_size)

    trainloader = get_dataloader(X_train, Y_train)
    testloader = get_dataloader(X_test, Y_test, batch_size=batch_size)
    return trainloader, testloader


def load_dataloader(dataset, batch_size=BATCH_SIZE):
    """ Function to create data loaders from dataset """
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=2)


def get_cifar10_dataset(dirpath="./data/CIFAR10"):
    """
    Get CIFAR 10 dataset
    Returns:
        [Tuple]: 
            (trainset, testset)
    """
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)
    cifar_stats = (
        CIFAR10_MEAN, CIFAR10_STD_DEV
    )  # from https://opacus.ai/tutorials/building_image_classifier

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32,
                                          padding=4,
                                          padding_mode='reflect'),
        torchvision.transforms.RandomHorizontalFlip(),
        #transforms.Resize(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(*cifar_stats, inplace=True)
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    valid_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(*cifar_stats)
    ])

    trainset = torchvision.datasets.CIFAR10(root=dirpath,
                                            train=True,
                                            download=True,
                                            transform=train_transforms)
    testset = torchvision.datasets.CIFAR10(root=dirpath,
                                           train=False,
                                           download=True,
                                           transform=valid_transforms)
    return trainset, testset


def load_cifar10_datasets_and_loaders(dirpath="./data/CIFAR10",
                                      batch_size=BATCH_SIZE):
    """
    Returns:
        [Tuple]: 
            (trainset, testset), (trainloader, testloader)
    """
    trainset, testset = get_cifar10_dataset(dirpath)
    trainloader, testloader = load_dataloader(
        trainset,
        batch_size=batch_size), load_dataloader(testset, batch_size=batch_size)

    return (trainset, testset), (trainloader, testloader)


def get_cifar100_dataset(dirpath="./data/CIFAR100", transforms=None):
    """Get CIFAR 100 dataset

    Args:
        dirpath (str, optional): _description_. Defaults to "./data/cifar100".
        transforms (_type_, optional): _description_. Defaults to None.
    """
    if transforms is None:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, ), (0.5, ), inplace=True),
        ])
    cifar100_train = torchvision.datasets.CIFAR100(dirpath,
                                                   train=True,
                                                   download=True,
                                                   transform=transforms)
    cifar100_test = torchvision.datasets.CIFAR100(dirpath,
                                                  train=False,
                                                  download=True,
                                                  transform=transforms)
    return cifar100_train, cifar100_test


def load_cifar100_datasets_and_loaders(dirpath="./data/CIFAR100",
                                       batch_size=BATCH_SIZE,
                                       transforms=None):
    """
    Returns:
        [Tuple]: 
            (trainset, testset), (trainloader, testloader)
    """
    trainset, testset = get_cifar100_dataset(dirpath, transforms=transforms)
    trainloader, testloader = load_dataloader(
        trainset,
        batch_size=batch_size), load_dataloader(testset, batch_size=batch_size)

    return (trainset, testset), (trainloader, testloader)


def get_mnist_dataset(dirpath="./data/MNIST"):
    """Get MNIST dataset

    Args:
        dirpath (str, optional): _description_. Defaults to "./data/MNIST".

    Returns:
        tuple : (trainset, testset)
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize((0.5, ), (0.5, ),inplace=True),
    ])

    trainset = torchvision.datasets.MNIST(dirpath,
                                          train=True,
                                          download=True,
                                          transform=transform)
    testset = torchvision.datasets.MNIST(dirpath,
                                         train=False,
                                         download=True,
                                         transform=transform)
    return trainset, testset


def load_mnist_datasets_and_loaders(dirpath="./data/MNIST",
                                    batch_size=BATCH_SIZE):
    """
    Returns:
        [Tuple]: 
            (trainset, testset), (trainloader, testloader)ransforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, ), (0.5, ), inplace=True),
    ])

    """

    trainset, testset = get_mnist_dataset(dirpath)
    trainloader, testloader = load_dataloader(
        trainset,
        batch_size=batch_size), load_dataloader(testset, batch_size=batch_size)
    return (trainset, testset), (trainloader, testloader)


def get_data_loaders(data_loader, num_teachers, batch_size=BATCH_SIZE):
    """ Function to create data loaders for the Teacher classifier """
    # data per teacher
    data_per_teacher = len(data_loader) // num_teachers

    return [
        torch.utils.data.DataLoader(Subset(
            data_loader,
            np.arange(0, data_per_teacher) + (data_per_teacher * teacher)),
                                    batch_size=batch_size)
        for teacher in range(num_teachers)
    ]


def dataset_from_indices(mask, dataset):
    """Generate a new dataset from a mask and a dataset.

    Args:
        mask (_type_): _description_
        dataset (_type_): _description_

    Raises:
        Exception: _description_

    Returns:
        dataset
    """
    indices = mask.nonzero()[0]
    if not (isinstance(indices, np.ndarray)):
        raise Exception(f"Indices are not numpy array: {type(indices)}")
    return Subset(dataset, indices)


def get_uneven_data_loaders(dataset, lengths):
    """ Function to create an uneven collection of dataloaders"""
    N = len(dataset)
    percent = int(N / 100)
    lengths = [
        percent * 30, percent * 20, percent * 15, percent * 10, percent * 5,
        percent * 5, percent * 5, percent * 5, percent * 5
    ]

    return torch.utils.data.random_split(dataset, lengths)


def get_dataset_by_name(dataset_name="mnist"):
    """ Get a dataset (tuple) by name (mnist, cifar10, cifar100)
    """
    if dataset_name.lower() == "mnist":
        print("Loading MNIST dataset")
        return get_mnist_dataset()
    elif dataset_name.lower() == "cifar10":
        print("Loading CIFAR10 dataset")
        return get_cifar10_dataset()
    elif dataset_name.lower() == "cifar100":
        print("Loading CIFAR100 dataset")
        return get_cifar100_dataset()

    raise Exception(f"Do not know name : {dataset_name}")


def get_dataset_and_dataloader_by_name(dataset_name):
    """ Get a dataset (tuple) by name (mnist, cifar10, cifar100)
    return 
        (trainset, testset), (trainloader, testloader)
    """
    trainset, testset = get_dataset_by_name(dataset_name)
    trainloader, testloader = load_dataloader(
        trainset,
        batch_size=BATCH_SIZE), load_dataloader(testset, batch_size=BATCH_SIZE)
    return (trainset, testset), (trainloader, testloader)
