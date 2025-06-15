"""All the in-depth ML training

Returns:
    _type_: _description_
"""
import datetime
import logging
import multiprocessing
import os
import warnings
from multiprocessing import Pool

import numpy as np
import torch
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from torch import optim
from tqdm import tqdm

from data_aware_dp import models, rdp, utils, sampling
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# linear regression dataset
RDP_ALPHA = 2


def get_criterion():
    return torch.nn.MSELoss()


def moving_average(x, w_):
    """moving average 

    Args:
        x (_type_): _description_
        w_ (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.convolve(x, np.ones(w_), 'valid') / w_


def convergence_value(losses, window=50):
    """
    Compute the convergence value for a given loss history.
    """
    losses_ = moving_average(np.array(losses), window)

    #convergence_index = np.argmin(losses_)
    convergence_index = -1
    conv_val = losses[convergence_index]
    #print(f"{conv_val}, {losses[0]}")

    if float(losses[0]) < float(conv_val):
        raise Exception(
            "Convergence value is greater than the first loss value.")
    return conv_val  # float(losses_[np.argmin(losses_)])


def time_to_convergence(losses, conv_val_window=5, window=1):
    """
    Compute the time to convergence for a given loss history.
    """
    smoothed_losses = moving_average(np.array(losses), window)
    conv_val = convergence_value(losses, window=conv_val_window)

    # within 3% of the conv value
    return np.argmin(smoothed_losses > conv_val * 1.03)


def train_one_epoch(*,
                    model,
                    optimizer,
                    train_loader,
                    criterion=torch.nn.MSELoss(),
                    verbose=False,
                    device=None,
                    categorical_data=False,
                    epoch):
    acc = None
    if device is None:
        device = models.get_default_device()
    total_loss = 0.

    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(inputs)

        # get loss for the predicted output
        loss = criterion(outputs.squeeze(), labels.squeeze())
        # get gradients w.r.t to parameters
        loss.backward()
        # update parameters
        optimizer.step()
        total_loss = +float(loss)
        if categorical_data:
            preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            labels = labels.detach().cpu().numpy()
            acc.append((preds == labels).mean())

    if verbose and epoch % 50 == 0:
        logging.debug("Epoch: {} | Loss: {}".format(epoch, loss.item()))
    return float(total_loss), float(np.mean(acc))


def train_one_epoch_privately(*,
                              model,
                              privacy_engine=None,
                              optimizer=None,
                              train_loader,
                              criterion=torch.nn.MSELoss(),
                              delta=1e-5,
                              MAX_PHYSICAL_BATCH_SIZE=128,
                              verbose=False,
                              device=None,
                              categorical_data=False,
                              scaler=None,
                              epoch):

    torch.cuda.empty_cache()

    if device is None:
        device = models.get_default_device()
    acc = []
    total_loss = 0.
    with BatchMemoryManager(data_loader=train_loader,
                            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
                            optimizer=optimizer) as memory_safe_data_loader:

        for i, (inputs, target) in enumerate(memory_safe_data_loader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            target = target.to(device)
            # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to acummulate gradients
            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    # get output from the model, given the inputs
                    outputs = model(inputs)
                    # get loss for the predicted output
                    loss = criterion(outputs.squeeze(), target.squeeze())
            outputs = model(inputs)
            # get loss for the predicted output
            loss = criterion(outputs.squeeze(), target.squeeze())
            # get gradients w.r.t to parameters
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            # update parameters
            if scaler is not None:  # mixed precision training
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            total_loss = +float(loss)

            if categorical_data:
                preds_ = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                target_ = target.detach().cpu().numpy()
                acc.append((preds_ == target_).mean())

    if verbose and epoch % 50 == 0:
        print("Epoch: {} | Loss: {}".format(epoch, total_loss))
    epsilon = None
    if privacy_engine is not None:
        epsilon = privacy_engine.get_epsilon(delta)

    return total_loss, epsilon, float(np.mean(acc))


def test(model, test_loader, device, criterion=None):
    model.eval()
    criterion = torch.nn.MSELoss() if criterion is None else criterion
    losses = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.squeeze())
            #loss = criterion(output, labels)
            losses.append(loss.item())

    return float(np.mean(losses))


def get_privacy_model(
    *,
    model,
    trainloader,
    epochs,
    target_epsilon,
    delta=1e-5,
    max_grad_norm,
    beta=None,
    LR=0.05,
    optimizer_factory=None,
    batch_size=128,
    device=None,
):
    """
    get a private model    

    Args:
        trainloader (_type_): _description_
        epochs (_type_): _description_
        epsilon (_type_): _description_
        max_grad_norm (_type_): _description_
        num_classes (int, optional): _description_. Defaults to 10.

    Returns:
        model, optimizer, train_loader
    """
    model = ModuleValidator.fix(model)
    errors = ModuleValidator.validate(model, strict=False)
    print(errors)

    sensitivity = max_grad_norm
    privacy_engine = PrivacyEngine()  # change this with Beta
    #model = get_base_model(num_classes=num_classes)
    N = len(trainloader.dataset)
    sample_rate = batch_size / N
    print(sample_rate)

    c = None
    beta_sampler = None

    if beta is not None:

        c = rdp.solve_for_c_val_for_equivalent_RDP_as_SGM(
            beta=beta,
            target_epsilon=target_epsilon,
            rdp_alpha=RDP_ALPHA,  # hardcoded
            delta=delta,
            sample_rate=sample_rate,  # 4096 / 45_000 = 9.1 %
            epochs=epochs,
            sensitivity=sensitivity)

        # this assumes we are using torch?
        beta_sampler = sampling.beta_exponential_sampler__torch(beta,
                                                                c,
                                                                device=device)
        #beta_sampler = sampling.inverse_transform_sampling_beta_exponential(beta, c)

    if optimizer_factory is None:
        optimizer_factory = optim.Adam

    optimizer = optimizer_factory(model.parameters(), lr=LR)

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=delta,
        max_grad_norm=max_grad_norm,
        #####
        beta_sampler=beta_sampler,
    )

    return model, optimizer, train_loader, privacy_engine, c


def train_model(
        *,
        model_factory,
        epochs,
        target_epsilon,
        device=None,
        trainloader,
        testloader=None,
        max_grad_norm=1.,
        criterion=None,
        delta=1e-5,
        batch_size=128,
        beta=None,
        categorical_data=True,
        optimizer_factory=None,  # optim.Adam,
        learning_rate_scheduler_f=None,
        LR=1e-3,
        input_dim=None,
        output_dim=None,
        verbose=True):
    """
    trains model privately or not privately
        depending on if epsilon = None

    categorical_data = True if data is categorical (i.e. mnist), False if regression

    returns 
        accuracies, model, (beta, c)
    """
    torch.cuda.empty_cache()

    if device is None:
        device = models.get_default_device()
    print(f"device is {device}")
    privacy_engine = None
    c = None
    if criterion is None:
        criterion = torch.nn.MSELoss()

    model = model_factory(
        input_dim=input_dim,
        output_dim=output_dim)  # get_lr_model(input_dim=input_dim)

    model.train()

    if target_epsilon is not None:
        logging.debug("get to priv model")

        model, optimizer, trainloader, privacy_engine, c = get_privacy_model(
            model=model,
            #input_dim=input_dim,
            trainloader=trainloader,
            epochs=epochs,
            target_epsilon=target_epsilon,
            delta=delta,
            max_grad_norm=max_grad_norm,
            #criterion = criterion,
            optimizer_factory=optimizer_factory,
            batch_size=batch_size,
            beta=beta,
            LR=LR,
            device=device)
        if verbose:
            logging.info(
                f"Using sigma={optimizer.noise_multiplier} and max_grad_norm={max_grad_norm}; c={c}"
            )

    else:
        optimizer = optim.Adam(model.parameters(), lr=LR)

        print("no privacy!")
    learning_rate_scheduler = None
    if learning_rate_scheduler_f is not None:
        learning_rate_scheduler = learning_rate_scheduler_f(optimizer)

    accuracies = []
    epsilons = []
    losses = []
    test_losses = []
    epoch_iterator_ = tqdm(range(epochs), desc="Epoch",
                           unit="epoch") if verbose else range(epochs)

    start = datetime.datetime.now()

    scaler = None  #  torch.cuda.amp.GradScaler() # mixed precision tool
    # currently getting: expected scalar type Half but found Float

    for epoch in epoch_iterator_:
        epoch_start = datetime.datetime.now()
        #acc,
        if privacy_engine is not None:
            loss, epsilon, acc = train_one_epoch_privately(
                model=model,
                train_loader=trainloader,
                optimizer=optimizer,
                epoch=epoch + 1,
                device=device,
                privacy_engine=privacy_engine,
                criterion=criterion,
                categorical_data=categorical_data,
                delta=delta,
                scaler=scaler,  # mixed precision tool
                verbose=verbose)
        else:
            loss, acc = train_one_epoch(model=model,
                                        train_loader=trainloader,
                                        optimizer=optimizer,
                                        epoch=epoch + 1,
                                        criterion=criterion,
                                        device=device,
                                        categorical_data=categorical_data,
                                        verbose=verbose)
            epsilon = None
        learning_rate_scheduler.step()
        print(f"epoch {epoch+1}/{epochs}")
        print(f"Time from first epoch: {datetime.datetime.now() - start}")
        print(f"Time from last epoch: {datetime.datetime.now() - start}")
        print(f"Loss : {loss}")
        if testloader is not None:
            test_loss = float(
                test(model=model,
                     test_loader=testloader,
                     device=device,
                     criterion=criterion))
            test_losses.append(test_loss)
            print(f"Test Loss : {test_loss}")
            model.train()
        if categorical_data:
            #logging.info(
            print(f"\ntrain accuracy: {round(acc * 100,2) }%")
        accuracies.append(acc)
        epsilons.append(epsilon)
        losses.append(loss)

    return accuracies, losses, test_losses, epsilons, model, (beta, c)


def do_a_single_eps_beta_run(args_eps):
    """for a given epsilon and beta, run n trials

        note: takes a single argument, so that it can be easily passed into `multiprocessing.Pool.map`

    Args:
        args (dictionary): 
            {
                "epsilon": eps,
                "trials": trials,

            }
    """
    ret = {"losses": [], "test_loss": [], "accuracies": [], "test_losses": []}
    args, eps = args_eps
    logging.debug("epsilon is" + str(eps))

    args["epsilon"] = eps
    trials = args.get("trials", 1)

    for _ in range(trials):
        beta = args["beta"]
        print(f"\t beta{beta}: epsilon {eps} ")

        print(f"model device is {args.get('device')}")
        try:
            accuracies, losses, test_losses, epsilon_history, model, (
                beta, c) = train_model(
                    model_factory=args.get("model_factory"),
                    epochs=args.get("epochs"),
                    target_epsilon=eps,  # 1.5, # 0.4,
                    device=args.get("device"),
                    trainloader=args.get("trainloader"),
                    testloader=args.get("testloader"),
                    beta=beta,
                    max_grad_norm=args.get("max_grad_norm", 1.),
                    delta=args.get("delta"),
                    categorical_data=args.get("categorical_data"),
                    batch_size=args.get("batch_size"),
                    criterion=args.get("criterion"),
                    optimizer_factory=args.get("optimizer_factory"),
                    learning_rate_scheduler_f=args.get(
                        "learning_rate_scheduler_f"),
                    LR=args.get("LR"),
                    input_dim=args.get("input_dim"),
                    output_dim=args.get("output_dim"),
                    verbose=args.get("verbose", False))
            test_loss = test(model,
                             args.get("testloader"),
                             args.get("device"),
                             criterion=args.get("criterion"))
            losses_ = [float(l) for l in losses]

            del model

            ret["losses"].append(losses_)
            ret["test_losses"].append(test_losses)
            ret["test_loss"].append(float(test_loss))
            ret["accuracies"].append(accuracies)
        except Exception as e:
            print(ret)
            print(f"exception:  {e}")
            continue
    del args

    return ret


def all_the_ml_training(*,
                        epsilons,
                        betas,
                        trainloader,
                        testloader,
                        save_path,
                        args,
                        categorical_data,
                        trials=1,
                        process_count=None,
                        do_multiprocessing=False):
    """Train an ML model for a range of epsilons and betas
        if running on a CPU, has the potential to run using multiple processes

    Args:
        epsilons (_type_): _description_
        betas (_type_): _description_
        trainloader (_type_): _description_
        testloader (_type_): _description_
        save_path (_type_): _description_
        args (_type_): _description_
        categorical_data (_type_): _description_
        trials (int, optional): _description_. Defaults to 1.
        process_count (_type_, optional): _description_. Defaults to None.
        do_multiprocessing (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    start = datetime.datetime.now()
    round_start = datetime.datetime.now()
    results = {}
    params = {
        "device": args.get("device"),
        "epochs": args.get("epochs"),
        "trainloader": trainloader,
        "testloader": testloader,
        "delta": args.get("delta", 1e-5),
        "categorical_data": categorical_data,
        "LR": args.get("LR"),
    }
    args.update(params)

    if process_count is None:
        process_count = multiprocessing.cpu_count() - 2,

    results = {}
    if os.path.exists(save_path):
        results = utils.open_yaml(save_path)

    for i, beta in enumerate(betas):
        args["beta"] = beta
        logging.debug(f"time from start {datetime.datetime.now() - start}")
        logging.debug(
            f"time from round start {datetime.datetime.now() - round_start}")
        round_start = datetime.datetime.now()
        beta = float(beta) if beta is not None else beta
        logging.info(beta)
        logging.debug(f"\n \n {i}/{len(betas)} \n\n ")

        if results.get(beta) is None:
            results[beta] = {}

        # for each beta, for each epsilon
        if not do_multiprocessing:
            for eps in epsilons:
                if results[beta].get(eps) is not None:
                    continue  #
                beta = float(beta) if beta is not None else None
                eps = float(eps)
                results[beta][eps] = do_a_single_eps_beta_run((args, eps))
                logging.debug(f"writing to {save_path}")
                #logging.debug(results)
                utils.write_yaml(results, save_path)

        else:
            args_trainf_eps_list = [(args, eps) for eps in epsilons]
            with Pool(processes=process_count) as pool:
                epsilon_results = pool.map(do_a_single_eps_beta_run,
                                           args_trainf_eps_list)

                for i, epsilon_result in enumerate(epsilon_results):
                    eps = epsilons[i]
                    results[beta][eps] = epsilon_result

        logging.debug(f"writing to {save_path}")
        #logging.debug(results)
        utils.write_yaml(results, save_path)

    return results
