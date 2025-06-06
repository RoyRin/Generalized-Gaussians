import os
import math

import numpy as np
import torch

ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


def scatter_normalization(train_loader,
                          scattering,
                          K,
                          device,
                          data_size,
                          sample_size,
                          noise_multiplier=1.0,
                          orders=ORDERS,
                          save_dir=None):
    # privately compute the mean and variance of scatternet features to normalize
    # the data.

    rdp = 0
    epsilon_norm = np.inf

    # try loading pre-computed stats
    use_scattering = scattering is not None
    assert use_scattering
    mean_path = os.path.join(
        save_dir,
        f"mean_bn_{sample_size}_{noise_multiplier}_{use_scattering}.npy")
    var_path = os.path.join(
        save_dir,
        f"var_bn_{sample_size}_{noise_multiplier}_{use_scattering}.npy")

    print(f"Using BN stats for {sample_size}/{data_size} samples")
    print(
        f"With noise_mul={noise_multiplier}, we get ε_norm = {epsilon_norm:.3f}"
    )

    try:
        print(f"loading {mean_path}")
        mean = np.load(mean_path)
        var = np.load(var_path)
        print(mean.shape, var.shape)
    except Exception as e:

        # compute the scattering transform and the mean and squared mean of features
        scatters = []
        mean = 0
        sq_mean = 0
        count = 0
        for idx, (data, target) in enumerate(train_loader):
            with torch.no_grad():
                data = data.to(device)
                if scattering is not None:
                    data = scattering(data).view(-1, K, data.shape[2] // 4,
                                                 data.shape[3] // 4)
                if noise_multiplier == 0:
                    data = data.reshape(len(data), K, -1).mean(-1)
                    mean += data.sum(0).cpu().numpy()
                    sq_mean += (data**2).sum(0).cpu().numpy()
                else:
                    scatters.append(data.cpu().numpy())

                count += len(data)
                if count >= sample_size:
                    break

        if noise_multiplier > 0:
            scatters = np.concatenate(scatters, axis=0)
            scatters = np.transpose(scatters, (0, 2, 3, 1))

            scatters = scatters[:sample_size]

            # s x K
            scatter_means = np.mean(scatters.reshape(len(scatters), -1, K),
                                    axis=1)
            norms = np.linalg.norm(scatter_means, axis=-1)

            # technically a small privacy leak, sue me...
            thresh_mean = np.quantile(norms, 0.5)
            scatter_means /= np.maximum(norms / thresh_mean, 1).reshape(-1, 1)
            mean = np.mean(scatter_means, axis=0)

            mean += np.random.normal(scale=thresh_mean * noise_multiplier,
                                     size=mean.shape) / sample_size

            # s x K
            scatter_sq_means = np.mean(
                (scatters**2).reshape(len(scatters), -1, K), axis=1)
            norms = np.linalg.norm(scatter_sq_means, axis=-1)

            # technically a small privacy leak, sue me...
            thresh_var = np.quantile(norms, 0.5)
            print(
                f"thresh_mean={thresh_mean:.2f}, thresh_var={thresh_var:.2f}")
            scatter_sq_means /= np.maximum(norms / thresh_var,
                                           1).reshape(-1, 1)
            sq_mean = np.mean(scatter_sq_means, axis=0)
            sq_mean += np.random.normal(scale=thresh_var * noise_multiplier,
                                        size=sq_mean.shape) / sample_size
            var = np.maximum(sq_mean - mean**2, 0)
        else:
            mean /= count
            sq_mean /= count
            var = np.maximum(sq_mean - mean**2, 0)

        if save_dir is not None:
            print(f"saving mean and var: {mean.shape} {var.shape}")
            np.save(mean_path, mean)
            np.save(var_path, var)

    mean = torch.from_numpy(mean).to(device)
    var = torch.from_numpy(var).to(device)

    return (mean, var), rdp
