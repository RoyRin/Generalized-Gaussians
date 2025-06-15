import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_aware_dp import sampling
#from opacus import privacy_engine  # library
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from torchvision import models

#from tqdm.notebook import tqdm
from opacus.accountants import prv
from opacus.accountants.analysis.prv import prvs

import local_models
import dataset_management
from data_aware_dp import models as dsdp_models

from tqdm import tqdm

import logging

warnings.simplefilter("ignore")

EPOCHS = 100
trials = 5

MAX_PHYSICAL_BATCH_SIZE = 1024  # 64


def accuracy(preds, labels):
    return (preds == labels).mean()


def train(
        model,
        train_loader,
        optimizer,
        epoch,
        device,
        privacy_engine_,
        #
        delta=1e-5,
        beta=None):
    model.train()
    torch.cuda.empty_cache()
    criterion = nn.CrossEntropyLoss()  # <<<
    torch.cuda.empty_cache()
    losses = []
    top1_acc = []
    print(f"get here")
    #print(model)

    with BatchMemoryManager(data_loader=train_loader,
                            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
                            optimizer=optimizer) as memory_safe_data_loader:
        torch.cuda.empty_cache()
        for i, (images, target) in enumerate(memory_safe_data_loader):
            print(f"i - {i}")
            optimizer.zero_grad()
            #images = images.detach().cpu().numpy().cuda()
            #target = target.detach().cpu().numpy().cuda()

            images = images.to(device)
            # target = F.one_hot(target)
            # target = target.to(torch.int64)
            target = target.to(device)

            # torch.float32
            # torch.int64

            # compute output
            output = model(images)
            # with torch.autocast('cuda'):
            #    loss = self.criterion(out, torch.tensor(labels).cuda())
            # print(target.dtype)
            # print(output.dtype)
            # print(criterion)
            #with torch.autocast('cuda'):
            loss = criterion(output, target)
            #print(loss)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)
            loss.backward()

            optimizer.step()  # add the noise to the gradients

            if (i + 1) % 200 == 0:
                # do privacy accounting
                print(i)
                epsilon = float(privacy_engine_.get_epsilon(delta, beta=beta))
                print(epsilon)
                print(f"Beta : {beta} - \tTrain Epoch: {epoch} \t"
                      f"Loss: {np.mean(losses):.6f} "
                      f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                      f"(ε = {epsilon:.2f}, δ = {delta })")

    epsilon = float(privacy_engine_.get_epsilon(delta, beta=beta))
    print(f"Beta : {beta} - \tTrain Epoch: {epoch} \t"
          f"Loss: {np.mean(losses):.6f} "
          f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
          f"(ε = {epsilon:.2f}, δ = {delta })")

    # HACK - NANMEAN
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



def __initialize_model(device, model_factory, LR=1e-2, momentum=0.9):
    print("initialize model")
    model = model_factory()
    model = ModuleValidator.fix(model)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=momentum)
    privacy_engine_ = PrivacyEngine(accountant="prv")
    return model, criterion, optimizer, privacy_engine_


def initialize_model(device, model, LR=1e-2, momentum=0.9):
    model = ModuleValidator.fix(model)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=momentum)
    privacy_engine_ = PrivacyEngine(accountant="prv")
    return model, criterion, optimizer, privacy_engine_


def initialize_model_old(device, LR=1e-2, momentum=0.9, **kwargs):
    model = models.resnet18(num_classes=10)
    model = dsdp_models.ResNet9(in_channels=3, num_classes=10)
    #model = models.resnet50(num_classes =10)
    model = ModuleValidator.fix(model)
    # ModuleValidator.validate(model, strict=False)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=momentum)
    privacy_engine_ = PrivacyEngine(accountant="prv")

    return model, criterion, optimizer, privacy_engine_


def initialize_model_resnet_18(device, LR=1e-2, momentum=0.9, **kwargs):
    model_factory = lambda: models.resnet18(num_classes=10)
    model = model_factory()
    return initialize_model(device, model, LR, momentum)


def initialize_model_resnet_9(device, LR=1e-2, momentum=0.9, **kwargs):
    model_factory = lambda: dsdp_models.ResNet9(in_channels=3, num_classes=10)
    model = model_factory()
    return initialize_model(device, model, LR, momentum)


def intialize_model_CNN(device, LR=1e-2, momentum=0.9, **kwargs):
    model_factory = lambda: local_models.CIFAR10_CNN(in_channel=3)
    model = model_factory()
    return initialize_model(device, model, LR, momentum)


def intialize_model_adult_CNN(device, LR=1e-2, momentum=0.9, **kwargs):
    model_factory = lambda: local_models.CNN_Adult()
    model = model_factory()
    return initialize_model(device, model, LR, momentum)


def intialize_model_LSTM(device, LR=1e-2, momentum=0.9, **kwargs):
    model_factory = lambda: local_models.LSTMNet()  # .cuda()
    model = model_factory()
    return initialize_model(device, model, LR, momentum)


import scatternet_cnns


def initialize_model_scatternet_cnns(device,
                                     train_loader,
                                     bn_noise_multiplier=8,
                                     LR=1e-2,
                                     momentum=0.9):
    model_factory = lambda: scatternet_cnns.get_scatternet_model(
        train_loader=train_loader,
        size=None,
        bn_noise_multiplier=bn_noise_multiplier,
        device=device)
    print("afterwards!")
    model = model_factory()
    return initialize_model(device, model, LR, momentum)


def initialize_model_WRN(device, LR=1e-2, momentum=0.9, **kwargs):
    model_factory = lambda: torch.hub.load(
        'pytorch/vision:v0.10.0',
        'wide_resnet50_2',
        pretrained=False  #, force_reload= True 
    )
    model = model_factory()
    return initialize_model(device, model, LR, momentum)


model_name_to_initializer = {
    "resnet18": initialize_model_resnet_18,
    "resnet9": initialize_model_resnet_9,
    "CNN": intialize_model_CNN,
    "scatternet_cnns": initialize_model_scatternet_cnns,
    "WRN": initialize_model_WRN,
    #
    "adult_FCN": intialize_model_adult_CNN,
    "LSTM": intialize_model_LSTM,
    # imdb
    # SNLI
}

import yaml
from pathlib import Path
import itertools
import sys


def write_yaml(d, path):
    with open(path, 'w') as outfile:
        yaml.dump(d, outfile, default_flow_style=False)


if __name__ == "__main__":
    index = int(sys.argv[1])
    hub_dir = Path(torch.hub.get_dir()) / f"checkpoints_{index}"
    torch.hub.set_dir(hub_dir)

    sigma_N = 6
    beta_N = 12
    sigmas = [round(s, 2) for s in np.linspace(.5, 3., sigma_N)]
    betas = list(
        set([None, 2.0] + [round(b, 2)
                           for b in np.linspace(1, 4, beta_N - 2)]))

    if index == -1:
        betas = [None]
    sigma_betas = list(itertools.product(sigmas, betas))
    sigma, beta = sigma_betas[index][0], sigma_betas[index][1]

    job_id = sys.argv[2] if len(sys.argv) > 2 else ""

    # WHAT WE ARE CURRENTLY ADDING.
    #dataset_name = "cifar-10"
    #model_name = "CNN"  # CNN # "scatter_net_CNN" # Handcrafted_cNN

    momentum = 0.9

    batch_sizes = [256, 512]
    #batch_sizes = [16]

    LRs = [.5, 1.]
    MAX_GRAD_NORMS = [.05, .1, .2, .4]
    delta_powers = [6]
    model_names = list(model_name_to_initializer.keys())

    dataset_to_model_names = {
        #"cifar-10": ["resnet18", "resnet9"], #"WRN", "scatternet_cnns", "CNN"
        #"cifar-10": ["WRN", "scatternet_cnns", "CNN"],
        #"svhn": ["WRN", "scatternet_cnns", "CNN"],
        "cifar-10": ["scatternet_cnns", "CNN"],
        "svhn": ["scatternet_cnns", "CNN"],
        #"svhn": ["resnet18", "resnet9", "CNN", "scatternet_cnns"], # "WRN"
        "adult": ["adult_FCN"],
        "imdb": ["LSTM"],
        # "snli": ["bert"],
        # "movielens": []
    }
    dataset_names = list(dataset_to_model_names.keys())
    dataset_names = ["imdb", "adult", "cifar-10", "svhn"]
    dataset_names = ["adult"]  #, "cifar-10", "svhn"]
    dataset_names = ["cifar-10"]  #, "cifar-10", "svhn"]
    dataset_names = ["svhn"]  #, "cifar-10", "svhn"]
    dataset_names = ["cifar-10", "svhn"]  #, "cifar-10", "svhn"]

    dataset_names = ["cifar-10", "svhn", "adult", "imdb"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_lr_norm_delta = list(
        itertools.product(batch_sizes, LRs, MAX_GRAD_NORMS, delta_powers))

    # dataset_names = ["imdb"]

    model_data_names = list(itertools.product(dataset_names, model_names))

    for batch_size, LR, max_grad_norm, delta_power in batch_lr_norm_delta:

        for dataset_name in dataset_names:
            print(
                f"{batch_size} - {LR} - {max_grad_norm} - {delta_power} -{dataset_name}"
            )
            data_root_dir = "".join(dataset_name.split("-")).upper()
            DATA_ROOT = f"../{data_root_dir}"

            if dataset_name == "svhn":
                train_loader, test_loader = dataset_management.initialize_data_SVHN(
                    batch_size=batch_size, DATA_ROOT=DATA_ROOT)
            elif dataset_name == "cifar-10":
                train_loader, test_loader = dataset_management.initialize_data_CIFAR10(
                    batch_size=batch_size, DATA_ROOT=DATA_ROOT)
            elif dataset_name == "adult":
                train_loader, test_loader = dataset_management.initialize_data_ADULT(
                    batch_size=batch_size, DATA_ROOT=DATA_ROOT)

            elif dataset_name == "imdb":
                train_loader, test_loader = dataset_management.initialize_data_IMDB(
                    batch_size=batch_size, DATA_ROOT=DATA_ROOT)

            model_names = dataset_to_model_names[dataset_name]
            for model_name in model_names:

                if model_name == "scatter_net_CNN":
                    train_loader, test_loader = scatternet_cnns.get_scatternet_loaders(
                        train_loader, test_loader, device=device)

                delta = 10**(-1 * delta_power)

                results = []
                sigma = float(sigma)
                scale = float(prvs.sigma_to_scale(sigma))

                save_path = f"{job_id}__{dataset_name}__experiment_{model_name}__beta_{beta}__sigma_{sigma}__batch_size_{batch_size}__LR_{LR}__max_grad_norm_{max_grad_norm}__delta_1e-{delta_power}"

                for trial in range(trials):
                    try:
                        #if True:
                        print(beta)
                        print(f"sigma is {sigma}")

                        logging.info(f"beta is {beta}")

                        # initialize_model

                        # THIS IS WHERE THE MAGIC IS.
                        model_initializer = model_name_to_initializer.get(
                            model_name)
                        if model_initializer is None:
                            print(
                                f"no model initializer found for {model_name}")
                            continue
                            raise Exception(
                                f"No model initializer found for model {model_name}"
                            )

                        model, criterion, optimizer, privacy_engine_ = model_initializer(
                            device=device,
                            train_loader=train_loader,
                            bn_noise_multiplier=8,
                            LR=1e-2,
                            momentum=0.9)

                        # we add noise using `scale`
                        # we do accounting use noise_mulitpler = sigma
                        # and then computing scale = sigma_to_scale (sigma )
                        print(f"sampling with beta {beta} and scale {scale}")
                        #beta_sampler = sampling.beta_exponential_sampler_from_scale(
                        #    beta=beta, scale=scale) if beta is not None else None

                        beta_sampler = sampling.beta_exponential_sampler__torch(
                            beta=beta,
                            scale=scale * max_grad_norm,
                            device=device) if beta is not None else None

                        model, optimizer, train_loader = privacy_engine_.make_private(
                            noise_multiplier=
                            sigma,  # note - the noise multiplier here, is what gets computed for the accounting
                            module=model,
                            optimizer=optimizer,
                            data_loader=train_loader,
                            #epochs=EPOCHS,
                            #target_epsilon=EPSILON,
                            # target_delta=delta ,
                            max_grad_norm=max_grad_norm,
                            #
                            beta=beta,
                            beta_sampler=beta_sampler)

                        accs, test_accs, losses, epsilons = [], [], [], []

                        scheduler = torch.optim.lr_scheduler.StepLR(
                            optimizer, step_size=10,
                            gamma=1.)  # 0.5) # every 10 steps, halve it

                        epochs = EPOCHS * 3 if model_name == "LSTM" else EPOCHS

                        for epoch in tqdm(range(epochs),
                                          desc="Epoch",
                                          unit="epoch"):
                            print(f"training on epoch {epoch}")
                            logging.info(f"training on epoch {epoch}")

                            top1_acc, loss, epsilon = train(
                                model,
                                train_loader,
                                optimizer,
                                epoch + 1,
                                device,
                                privacy_engine_=privacy_engine_,
                                delta=delta,
                                beta=beta)
                            print(f"scheduler")
                            print(scheduler.get_last_lr())
                            scheduler.step()
                            print(f"completed training epoch ")

                            accs.append(float(top1_acc))
                            losses.append(float(loss))
                            epsilons.append(epsilon)
                            print(f"acc : {top1_acc}")
                            logging.info(top1_acc)

                            top1_acc__test = float(
                                test(model, test_loader, device))
                            test_accs.append(top1_acc__test)

                            single_run = {
                                "train_acc": accs,
                                "test_acc": test_accs,
                                "train_loss": losses,
                                "epsilon": epsilons,
                                #"test_acc": top1_acc
                                "sigma": sigma,
                                "scale": scale,
                                "beta":
                                float(beta) if beta is not None else None,
                                "LR": LR,
                                "batch_size": batch_size,
                                "model_name": model_name,
                                "momentum": momentum,
                                "max_grad_norm": max_grad_norm,
                                "delta": delta,
                                "dataset_name": dataset_name
                            }

                            write_yaml(single_run, f"{save_path}__temp.yaml")
                            if epsilon > 10.:
                                break

                        single_run = {
                            "train_acc": accs,
                            "train_loss": losses,
                            "test_acc": test_accs,
                            "epsilons": epsilons,
                            "sigma": sigma,
                            "scale": scale,
                            "beta": float(beta) if beta is not None else None,
                            "LR": LR,
                            "batch_size": batch_size,
                            "model_name": model_name,
                            "momentum": momentum,
                            "max_grad_norm": max_grad_norm,
                            "delta": delta,
                            "dataset_name": dataset_name
                        }
                        results.append(single_run)
                        write_yaml(results, f"{save_path}.yaml")

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
                    except Exception as e:
                        print(f"Error: {e}")
                        continue
