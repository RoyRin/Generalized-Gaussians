import numpy as np
from pathlib import Path
from importlib import reload
import data_aware_dp
from data_aware_dp import histogram_computation, evaluation, sampling
import datetime
import warnings
import itertools
import tqdm
import multiprocess as mp

cwd = Path.cwd()
labels = cwd / "multiple-teachers-for-privacy/"
# list files in labels
list(labels.glob('*npy'))

file_data = {
    f.stem: np.load(f, allow_pickle=True)
    for f in labels.glob('*npy')
}

# get mnist dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATASET = "SVHN"
DATASET = "MNIST"

mnist_train = datasets.MNIST(root='data',
                             train=True,
                             download=True,
                             transform=transforms.ToTensor())
mnist_test = datasets.MNIST(root='data',
                            train=False,
                            download=True,
                            transform=transforms.ToTensor())

svhn_train = datasets.SVHN(root='data',
                           split='train',
                           download=True,
                           transform=transforms.ToTensor())
svhn_test = datasets.SVHN(root='data',
                          split='test',
                          download=True,
                          transform=transforms.ToTensor())

if DATASET == "MNIST":
    teach_preds = file_data["mnist_250_teachers_labels"]
    PATE_labels = mnist_test.test_labels[:
                                         9000]  #P # based on comparing to the argmaxes, this is correct

elif DATASET == "SVHN":
    teach_preds = file_data["svhn_250_teachers_labels"]
    PATE_labels = svhn_test.labels  #P # based on comparing to the argmaxes, this is correct

# take argmax
#teach_preds = np.argmax(teach_preds, axis=0)


def get_histogram(votes):
    return np.array([np.sum(votes == i) for i in range(10)])


def compute_argmax(votes):
    return np.argmax([np.sum(votes == i) for i in range(10)])


teach_preds_histograms = np.array(
    [get_histogram(teach_preds[:, i]) for i in range(teach_preds.shape[1])])

teach_preds_argmax = np.array(
    [compute_argmax(teach_preds[:, i]) for i in range(teach_preds.shape[1])])
print(teach_preds_argmax.shape)
pred_len = len(teach_preds_argmax)
#N = len(mnist_test.test_labels)

correct_labels = PATE_labels


def get_noised_argmax_accuracy(histogram, correct_label, sampler, trials=100):
    """ given a histogram, get the accuracy of the argmax of the noisy histogram

    Args:
        histogram (_type_): _description_
        noise_source (_type_): _description_
        trials (_type_): _description_
        correct_label (_type_): _description_

    Returns:
        _type_: _description_
    """
    #return np.average([np.argmax(histogram + sampler(len(histogram))) == correct_label for _ in range(trials)])
    correct = 0
    for i in range(trials):
        noised_label = np.argmax(histogram + sampler(len(histogram)))
        correct += int(noised_label == correct_label)
    return correct / trials


def get_noised_argmax_accuracy_and_std(histograms,
                                       labels,
                                       sampler,
                                       trials=100):
    accs = []
    for i in range(trials):
        noised_histograms = histograms + sampler(histograms.shape)
        votes = np.argmax(noised_histograms, axis=1)
        correct = np.sum(np.array(votes) == np.array(labels))
        accs.append(correct / len(labels))
    acc = np.average(accs)
    std = np.std(accs)
    return acc, std


classes = 10
trials = 100

total_scale_count, beta_count = 401, 40
scales = np.linspace(.1, 100.1, total_scale_count)
scales = np.array([round(x, 2) for x in scales])
betas = [round(x, 2) for x in np.linspace(1, 4., beta_count)]

scale_betas = list(itertools.product(scales, betas))

scale_betas_enumerated = [(i, tup) for i, tup in enumerate(scale_betas)]

COUNTER = 0

results = []


def progress(results):
    print('.', end='', flush=True)


def compute_acc(tup):
    global COUNTER
    COUNTER += 1
    if COUNTER % 50 == 0:
        print(f"Counter : {COUNTER}")
    i, (scale, beta) = tup

    sampler = sampling.beta_exponential_sampler_from_scale(beta=beta,
                                                           scale=scale)
    if False:
        acc = np.average(
            np.array([
                get_noised_argmax_accuracy(teach_preds_histograms[i],
                                           PATE_labels[i],
                                           sampler,
                                           trials=trials)
                for i in range(len(PATE_labels))
            ]))
    acc, std = get_noised_argmax_accuracy_and_std(teach_preds_histograms,
                                                  PATE_labels,
                                                  sampler=sampler,
                                                  trials=trials)
    ret = [float(acc), float(std), float(scale), float(beta)]
    return ret
    print(ret)
    print(results)
    results.append(ret)


print(f"Total count : {len(scale_betas)}")
starttime = datetime.datetime.now()

cpu_count = mp.cpu_count()
with mp.Pool(cpu_count - 1) as pool:
    results = list(
        tqdm.tqdm(pool.imap(compute_acc, scale_betas_enumerated),
                  total=len(scale_betas_enumerated)))

    #results = p.map(compute_acc, scale_betas_enumerated)
    #results =p.apply_async(compute_acc, scale_betas_enumerated, callback=progress)
endtime = datetime.datetime.now()
print(f"Total time taken : {endtime - starttime}")
#pool = mp.Pool(12)
#for _ in tqdm.tqdm(pool.imap_unordered(compute_acc, scale_betas_enumerated), total=len(scale_betas_enumerated)):
#    pass

# save results in save path as yaml
import yaml
import datetime

start = datetime.datetime.now()
date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
save_fn = f"PATE__{DATASET}__classes__{classes}__beta_exponential_rdp_accuracy__{date_str}.csv"

print(f"Saving file in {save_fn}")
with open(save_fn, 'w') as f:
    yaml.dump(results, f)
