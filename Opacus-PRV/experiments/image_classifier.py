import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from data_aware_dp import sampling
#from opacus import privacy_engine  # library
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from torchvision import models
from torchvision.datasets import CIFAR10
#from tqdm.notebook import tqdm
from opacus.accountants import prv
from opacus.accountants.analysis.prv import prvs

from tqdm import tqdm

import logging

# These values, specific to the CIFAR10 dataset, are assumed to be known.
# If necessary, they can be computed with modest privacy budget.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
])

warnings.simplefilter("ignore")

MAX_GRAD_NORM = 1.2
EPSILON = 50.0
DELTA = 1e-5
EPOCHS = 20
LR = 1e-3

BATCH_SIZE = 128
MAX_PHYSICAL_BATCH_SIZE = 32

DATA_ROOT = '../cifar10'


def accuracy(preds, labels):
    return (preds == labels).mean()


def train(model, train_loader, optimizer, epoch, device, privacy_engine_):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    with BatchMemoryManager(data_loader=train_loader,
                            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
                            optimizer=optimizer) as memory_safe_data_loader:
        torch.cuda.empty_cache()
        for i, (images, target) in enumerate(memory_safe_data_loader):
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

            loss.backward()
            optimizer.step()

            if (i + 1) % 200 == 0:
                epsilon = privacy_engine_.get_epsilon(DELTA)
                print(f"\tTrain Epoch: {epoch} \t"
                      f"Loss: {np.mean(losses):.6f} "
                      f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                      f"(ε = {epsilon:.2f}, δ = {DELTA})")
    return np.mean(top1_acc), np.mean(losses), epsilon


def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

    top1_avg = np.mean(top1_acc)

    print(f"\tTest set:"
          f"Loss: {np.mean(losses):.6f} "
          f"Acc: {top1_avg * 100:.6f} ")
    return np.mean(top1_acc)


def sigma_to_scale(sigma):
    return sigma * np.sqrt(2)


def initialize_data():

    train_dataset = CIFAR10(root=DATA_ROOT,
                            train=True,
                            download=True,
                            transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
    )

    test_dataset = CIFAR10(root=DATA_ROOT,
                           train=False,
                           download=True,
                           transform=transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return train_loader, test_loader


def initialize_model(device):
    model = models.resnet18(num_classes=10)
    model = ModuleValidator.fix(model)
    # ModuleValidator.validate(model, strict=False)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=LR)
    privacy_engine_ = PrivacyEngine()  # accountant="prv")

    return model, criterion, optimizer, privacy_engine_


import yaml


def write_yaml(d, path):
    with open(path, 'w') as outfile:
        yaml.dump(d, outfile, default_flow_style=False)


if __name__ == "__main__":
    trials = 3
    # take CLI argument
    import sys
    train_loader, test_loader = initialize_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # random value
    #sigma = 0.39066894531249996

    sigmas = [0.3, 1, 5]
    #scale = sigma_to_scale(sigma)

    #betas = np.linspace(1., 3, 10)
    betas = [None]

    results = []

    for sigma in sigmas:
        sigma = float(sigma)
        scale = float(sigma_to_scale(sigma))

        for beta in betas:
            for trial in range(trials):
                print(beta)
                print(f"sigma is {sigma}")

                logging.info(f"beta is {beta}")

                model, criterion, optimizer, privacy_engine_ = initialize_model(
                    device=device)

                # we add noise using `scale`
                # we do accounting use noise_mulitpler = sigma
                # and then computing scale = sigma_to_scale (sigma )

                model, optimizer, train_loader = privacy_engine_.make_private(
                    noise_multiplier=
                    sigma,  # note - the noise multiplier here, is what gets computed for the accounting
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    #epochs=EPOCHS,
                    #target_epsilon=EPSILON,
                    #target_delta=DELTA,
                    max_grad_norm=MAX_GRAD_NORM,
                    #
                    # beta=beta,
                    # beta_sampler=beta_sampler
                )
                print(
                    f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}"
                )

                accs, losses, epsilons = [], [], []

                for epoch in tqdm(range(EPOCHS), desc="Epoch", unit="epoch"):
                    print(f"training on epoch {epoch}")
                    logging.info(f"training on epoch {epoch}")

                    top1_acc, loss, epsilon = train(
                        model,
                        train_loader,
                        optimizer,
                        epoch + 1,
                        device,
                        privacy_engine_=privacy_engine_,
                        # beta=beta
                    )
                    accs.append(float(top1_acc))
                    losses.append(float(loss))
                    epsilons.append(epsilon)
                    print(f"acc : {top1_acc}")
                    logging.info(top1_acc)
                    single_run = {
                        "beta": beta,
                        "train_acc": accs,
                        "train_loss": losses,
                        "epsilon": epsilons,
                        #"test_acc": top1_acc
                        "sigma": sigma,
                        "scale": scale
                    }

                    write_yaml(single_run, f"beta_{beta}__temp.yaml")

                top1_acc = float(test(model, test_loader, device))
                single_run = {
                    "beta": beta,
                    "train_acc": accs,
                    "train_loss": losses,
                    "test_acc": top1_acc,
                    "epsilons": epsilons,
                    "sigma": sigma,
                    "scale": scale,
                }
                results.append(single_run)
                write_yaml(results, f"beta_{beta}.yaml")

    #sample_rate = 1 / len(train_loader)
    #for beta in betas:
    #    print(f"beta is {beta}")
    #    sigma = privacy_engine.get_noise_multiplier(
    #        target_epsilon=EPSILON,
    #        target_delta=DELTA,
    #        sample_rate=sample_rate,
    #        epochs=EPOCHS,
    #        accountant="prv",
    #        #
    #        beta=beta)

    #    scale = sigma_to_scale(sigma)
