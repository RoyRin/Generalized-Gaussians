from torchvision.datasets import CIFAR10, SVHN
import torch
import torchvision
import torchvision.transforms as transforms

import tensorflow as tf
from keras.preprocessing import sequence
import numpy as np

# These values, specific to the CIFAR10 dataset, are assumed to be known.
# If necessary, they can be computed with modest privacy budget.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

SVHN_MEAN = (0.4376821, 0.4437697, 0.47280442)
SVHN_STD_DEV = (0.19803012, 0.20101562, 0.19703614)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
])

transform_SVHN = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD_DEV),
])


def initialize_data_CIFAR10(batch_size=1024, DATA_ROOT="../cifar10"):
    train_dataset = CIFAR10(root=DATA_ROOT,
                            train=True,
                            download=True,
                            transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
    )

    test_dataset = CIFAR10(root=DATA_ROOT,
                           train=False,
                           download=True,
                           transform=transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


def initialize_data_SVHN(batch_size=1024, DATA_ROOT="../SVHN"):
    train_dataset = SVHN(root=DATA_ROOT,
                         split='train',
                         download=True,
                         transform=transform_SVHN)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
    )

    test_dataset = SVHN(root=DATA_ROOT,
                        split='test',
                        download=True,
                        transform=transform_SVHN)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


import pandas as pd

from sklearn.model_selection import train_test_split


# custom dataset
class AdultDataset(torch.utils.data.Dataset):

    def __init__(self, images, labels):
        self.X = images
        self.y = labels

    def __len__(self):
        return (self.X.shape[0])

    def __getitem__(self, i):
        data = self.X.iloc[i, :]
        return (data.to_numpy(), self.y[i])


def initialize_data_ADULT(batch_size=1024, DATA_ROOT="../ADULT"):
    dir_path = "/h/royrin/new_code/data-aware-dp/Opacus-PRV/experiments"
    path = f"{dir_path}/adult.csv"
    x = pd.read_csv(path)  # 'adult.csv')
    trainData, testData = train_test_split(x, test_size=0.1, random_state=218)
    # have to reset index, see https://discuss.pytorch.org/t/keyerror-when-enumerating-over-dataloader/54210/13
    trainData = trainData.reset_index()
    testData = testData.reset_index()

    train_data = trainData.iloc[:, 1:-1].astype('float32')
    test_data = testData.iloc[:, 1:-1].astype('float32')
    # train_labels = (trainData.iloc[:, -1] == 1).astype('int32')
    # test_labels = (testData.iloc[:, -1] == 1).astype('int32') # targets = targets.type(torch.LongTensor)
    train_labels = (trainData.iloc[:, -1] == 1).astype('int64')
    test_labels = (testData.iloc[:, -1] == 1).astype('int64')

    kwargs = {"num_workers": 1, "pin_memory": True}

    train_loader = torch.utils.data.DataLoader(AdultDataset(
        train_data, train_labels),
                                               batch_size=batch_size,
                                               shuffle=True,
                                               **kwargs)

    test_loader = torch.utils.data.DataLoader(AdultDataset(
        test_data, test_labels),
                                              batch_size=batch_size,
                                              shuffle=True,
                                              **kwargs)
    return train_loader, test_loader


class IMDBDataset(torch.utils.data.Dataset):

    def __init__(self, images, labels):
        self.X = images
        self.y = labels

    def __len__(self):
        return (self.X.shape[0])

    def __getitem__(self, i):
        #data = self.X[i] # .iloc[i, :]
        return (self.X[i], self.y[i])


def initialize_data_IMDB(batch_size=1024,
                         DATA_ROOT="../imdb",
                         max_features=10_000,
                         max_len=256,
                         **kwargs):
    """Load IMDB movie reviews data."""
    train, test = tf.keras.datasets.imdb.load_data(num_words=max_features)
    (train_data, train_labels), (test_data, test_labels) = train, test

    train_data = sequence.pad_sequences(train_data,
                                        maxlen=max_len).astype(np.int32)
    test_data = sequence.pad_sequences(test_data,
                                       maxlen=max_len).astype(np.int32)
    train_labels, test_labels = train_labels.astype(
        np.int64), test_labels.astype(np.int64)
    #return (train_data, train_labels), (test_data, test_labels)
    # test_dataset = SVHN(root=DATA_ROOT,
    #                     split='test',
    #                     download=True,
    #                     transform=transform_SVHN)

    train_loader = torch.utils.data.DataLoader(IMDBDataset(
        train_data, train_labels),
                                               batch_size=batch_size,
                                               shuffle=True,
                                               **kwargs)

    test_loader = torch.utils.data.DataLoader(IMDBDataset(
        test_data, test_labels),
                                              batch_size=batch_size,
                                              shuffle=True,
                                              **kwargs)
    return train_loader, test_loader


if False:
    # import datasets
    from torchtext.datasets import IMDB
    train_iter = IMDB(split='train')

    def tokenize(label, line):
        return line.split()

    tokens = []
    for label, line in train_iter:
        tokens += tokenize(label, line)
