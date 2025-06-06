"""Wrappers for training ML models, and the helper functions

"""
import torch

from data_aware_dp import datasets, models, ml_utils
from torch import optim
import warnings
import logging

warnings.filterwarnings("ignore")


def mnist_model_factory(input_dim=None, output_dim=None):
    return models.CNN_model_factory(num_classes=10,
                                    device=models.get_default_device(),
                                    seed=None)


def mnist_training(args, batch_size, betas, epsilons, save_path, trials,
                   epochs):
    _, (mnist_trainloader,
        mnist_testloader) = datasets.load_mnist_datasets_and_loaders(
            batch_size=batch_size)

    args["model_factory"] = mnist_model_factory
    args["categorical_data"] = True
    args["optimizer_factory"] = optim.SGD  # optim.Adam #

    return ml_utils.all_the_ml_training(
        epsilons=epsilons,
        betas=betas,
        trainloader=mnist_trainloader,
        testloader=mnist_testloader,
        save_path=save_path,
        categorical_data=True,
        args=args,
    )


def cifar10_wide_model_factory(input_dim=None, output_dim=None):
    #return models.Resnet50_model_factory( num_classes=10, device=models.get_default_device())
    return models.get_cifar10_wide_resnet(num_classes=10,
                                          cuda=True,
                                          seed=None,
                                          pretrained=False)


def cifar10_wide_model_factory__2(input_dim=None, output_dim=None):

    return models.get_cifar10_wide_resnet(num_classes=10,
                                          cuda=True,
                                          seed=None,
                                          pretrained=False)


def cifar10_model_factory(input_dim=None, output_dim=None):
    #return models.Resnet50_model_factory( num_classes=10, device=models.get_default_device())
    return models.get_resnet18_privacy_model(num_classes=10,
                                             cuda=True,
                                             seed=None)

    return models.cifar10_CNN_model_factory(num_classes=10,
                                            device=models.get_default_device(),
                                            seed=None)


def cifar10_training(args,
                     batch_size,
                     betas,
                     epsilons,
                     save_path,
                     trials,
                     epochs,
                     wide_net=False):
    _, (cifar10_trainloader,
        cifar10_testloader) = datasets.load_cifar10_datasets_and_loaders(
            batch_size=batch_size)

    args[
        "model_factory"] = cifar10_wide_model_factory__2 if wide_net else cifar10_model_factory

    #cifar10_wide_model_factory

    args["categorical_data"] = True

    args["optimizer_factory"] = optim.SGD  # RMSprop  #optim.Adam
    args["delta"] = 1e-5
    logging.debug(args)

    return ml_utils.all_the_ml_training(epsilons=epsilons,
                                        betas=betas,
                                        trainloader=cifar10_trainloader,
                                        testloader=cifar10_testloader,
                                        save_path=save_path,
                                        args=args,
                                        categorical_data=True)


def lr_model_factory(input_dim=None, output_dim=None):
    return models.get_lr_model(input_dim=input_dim, cuda=False)


def linear_regression_training(args,
                               batch_size,
                               betas,
                               epsilons,
                               save_path,
                               process_count=None,
                               n_samples=1000,
                               noise=0):
    input_dim = args["input_dim"]

    trainloader, testloader = datasets.get_LR_dataloader(n_samples=n_samples,
                                                         input_dim=input_dim,
                                                         noise=noise,
                                                         batch_size=batch_size)

    input_dim = args["input_dim"]
    args["device"] = torch.device("cpu")
    args["model_factory"] = lr_model_factory
    args["categorical_data"] = False
    args["optimizer_factory"] = optim.Adam

    args["delta"] = 1e-5

    return ml_utils.all_the_ml_training(epsilons=epsilons,
                                        betas=betas,
                                        trainloader=trainloader,
                                        testloader=testloader,
                                        save_path=save_path,
                                        args=args,
                                        categorical_data=False,
                                        process_count=process_count,
                                        do_multiprocessing=False)  # True)
