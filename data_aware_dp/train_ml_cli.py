import sys

import numpy as np
import datetime
from data_aware_dp import models, ml_training, utils
import torch
import logging
import click
import os

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
format = "%(asctime)s - %(levelname)s - %(message)s"
format = "%(levelname)s - %(message)s"
stream_handler.setFormatter(
    logging.Formatter(format, datefmt="%Y-%m-%d %H:%M:%S"))

logging.basicConfig(level=logging.INFO, handlers=[stream_handler])
logger = logging.getLogger()
logger.setLevel(logging.WARNING)


@click.group(help="""CLI to run DP-SGD training for data-aware-dp""")
@click.pass_context
def cli(ctx):
    return


def get_betas(beta_count=40):
    s = int(beta_count / 4)  #
    betas = list(np.linspace(1, 2.5, 3 * s)) + list(np.linspace(2.6, 4, s))

    betas = [float(beta) for beta in betas] + [None]
    return betas


@cli.command(
    name="linear-regression",
    help="""run linear regression with DP-SGD, on an artificial dataset""")
@click.option("-s", "--save_dir", "-s", default=".")
@click.option("-c", "--epsilon-count", default=30, help="number of epsilons")
@click.option("--epsilon-min", default=.5, help=" min epsilon value")
@click.option("--epsilon-max", default=.5, help="max epsilon value")
@click.option("-b",
              "--beta-count",
              default=10,
              type=int,
              help="max epsilon value")
@click.option("-p",
              "--process-count",
              type=int,
              default=None,
              help="Number of processes to use")
@click.option("-n",
              "--noise",
              type=int,
              default=None,
              help="amount of noise to add")
@click.option("-v", "--verbose", is_flag=True, default=True)
@click.pass_context
def linear_regression(ctx, save_dir, epsilon_count, epsilon_min, epsilon_max,
                      process_count, beta_count, noise, verbose):

    max_grad_norm = 1.
    delta = 1e-5
    learning_rate = 2.

    learning_rate_scheduler_f = lambda optimizer_: torch.optim.lr_scheduler.StepLR(
        optimizer_, step_size=50, gamma=0.5)  # scales by 2 every 20 steps

    batch_size = 256
    epochs = 500
    trials = 10
    input_dim = 5
    n_samples = 1000

    device = torch.device('cpu')
    #device = torch.device('cuda')
    criterion = torch.nn.MSELoss()

    epsilons = [
        float(eps)
        for eps in np.linspace(epsilon_min, epsilon_max, epsilon_count)
    ]

    #betas = get_betas(beta_count)
    betas = list(np.linspace(1.5, 2.5, beta_count))
    betas = list(np.linspace(1, 4, beta_count))
    betas = [
        round((1 + (b // 0.02)) * 0.02, 2) for b in betas
    ]  # round beta to the nearest 0.02 (for ease of RDP val pre-computation)
    betas = list(
        set(betas + [1., 2.])
    )  # remove duplicates, and make sure gaussian and laplace are included

    print(betas)

    betas = [float(beta) for beta in betas] + [None]
    args = {
        "device": device,
        "max_grad_norm": max_grad_norm,
        "batch_size": batch_size,
        "input_dim": input_dim,
        "criterion": criterion,
        "verbose": verbose,
        "trials": trials,
        "epochs": epochs,
        "LR": learning_rate,
        "delta": delta,
        "learning_rate_scheduler_f": learning_rate_scheduler_f
    }

    logging.info(f"doing linear regression")

    if noise is None:
        noises = [2000, 100, 0]
    else:
        noises = [noise]

    for noise in noises:
        logging.info(f"noise: {noise}")

        save_path = os.path.join(
            save_dir,
            f"ML_LR_results_-1_noise_{noise}__{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.yaml"
        )

        ml_training.linear_regression_training(args,
                                               batch_size,
                                               betas,
                                               epsilons,
                                               save_path,
                                               process_count=process_count,
                                               n_samples=n_samples,
                                               noise=noise)


@cli.command(name="cifar-training",
             help="""run ML training with DP-SGD on CIFAR-10 dataset""")
@click.option(
    "-s",
    "--save_path",
    "-s",
    default=
    f"ML_cifar_results_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.yaml"
)
@click.option(
    "-g",
    "--group-index",
    default=None,
    help="which section of the betas to run (for processing on multiple GPUs)")
@click.option("-c", "--config-path", required=True, help="Config file path")
@click.option("-v", "--verbose", is_flag=True, default=True, help="Verbose")
@click.pass_context
def cifar_training(ctx, save_path, group_index, config_path, verbose):

    config = utils.open_yaml(config_path)

    num_of_groups = config.get("num_of_groups", 1)
    epsilon_min = config.get("epsilon_min", 0.1)
    epsilon_max = config.get("epsilon_max", 4)
    epsilon_count = config.get("epsilon_count", 30)
    beta_count = config.get("beta_count", 30)
    epochs = config.get("epochs", 30)
    batch_size = config.get("batch_size", 512)  # batch_size = 2 ** 11 # 2048
    trials = config.get("trials", 1)
    LR = config.get("LR", 1e-3)
    wide_net = config.get("wide_net", False)

    max_grad_norm = 1.
    delta = 1e-5

    device = models.get_default_device()
    criterion = torch.nn.CrossEntropyLoss()  # }MSELoss()

    epsilons = [
        float(eps)
        for eps in np.linspace(epsilon_min, epsilon_max, epsilon_count)
    ]

    betas = get_betas(beta_count)

    if (group_index is not None) and (num_of_groups is not None):
        num_of_groups = int(num_of_groups)
        group_index = int(group_index)
        section_size = int(beta_count / num_of_groups)
        print(f"total betas: {betas}")
        betas = betas[group_index * section_size:(group_index + 1) *
                      section_size]
        print(f"running on betas {betas}")

    args = {
        "device": device,
        "max_grad_norm": max_grad_norm,
        "batch_size": batch_size,
        "input_dim": 1,  # filler
        "criterion": criterion,
        "verbose": verbose,
        "trials": trials,
        "epochs": epochs,
        "LR": LR,
        "delta": delta
    }
    logging.info(f"doing cifar")

    ml_training.cifar10_training(args,
                                 batch_size,
                                 betas,
                                 epsilons,
                                 save_path,
                                 trials,
                                 epochs=epochs,
                                 wide_net=wide_net)


@cli.command(name="mnist-training",
             help="""run ML training with DP-SGD on MNIST dataset""")
@click.option(
    "-s",
    "--save_path",
    "-s",
    default=
    f"ML_mnist_results_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.yaml"
)
@click.option(
    "-g",
    "--group-index",
    default=None,
    help="which section of the betas to run (for processing on multiple GPUs)")
@click.option("-c", "--config-path", required=True, help="Config file path")
@click.option("-v", "--verbose", is_flag=True, default=True)
@click.pass_context
def mnist_training(ctx, save_path, group_index, config_path, verbose):
    config = utils.open_yaml(config_path)

    num_of_groups = config.get("num_of_groups", 1)
    epsilon_min = config.get("epsilon_min", 0.1)
    epsilon_max = config.get("epsilon_max", 4)
    epsilon_count = config.get("epsilon_count", 30)
    beta_count = config.get("beta_count", 30)

    max_grad_norm = 1.
    delta = 1e-5

    device = models.get_default_device()
    criterion = torch.nn.CrossEntropyLoss()  # }MSELoss()
    print(f"the device is {device}")

    epsilons = [
        float(eps)
        for eps in np.linspace(epsilon_min, epsilon_max, epsilon_count)
    ]

    betas = get_betas(beta_count)

    if (group_index is not None) and (num_of_groups is not None):
        num_of_groups = int(num_of_groups)
        group_index = int(group_index)
        section_size = int(beta_count / num_of_groups)
        print(f"total betas: {betas}")
        betas = betas[group_index * section_size:(group_index + 1) *
                      section_size]
        print(f"running on betas {betas}")

    args = {
        "device": device,
        "max_grad_norm": max_grad_norm,
        "batch_size": config.get("batch_size", 512),
        "input_dim": 4,  # filler
        "criterion": criterion,
        "verbose": verbose,
        "trials": config.get("trials", 1),
        "epochs": config.get("epochs", 30),
        "LR": config.get("LR", 1e-3),
        "delta": delta
    }
    logging.info(f"doing mnist")

    ml_training.mnist_training(args,
                               config.get("batch_size", 512),
                               betas,
                               epsilons,
                               save_path,
                               config.get("trials", 512),
                               epochs=config.get("epochs", 30))


if __name__ == "__main__":
    cli()
