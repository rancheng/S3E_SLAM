import numpy as np
import random
import time
import argparse

import torch
from torch.utils.data.sampler import Sampler
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import MinkowskiEngine as ME

def image_coords_ops(img_data, K):
    img_h, img_w = img_data.shape
    nx, ny = (img_h, img_w)
    x = np.linspace(0, nx, nx)
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    xv, yv = np.meshgrid(x, y)
    coords_uv = np.dstack((xv, yv)).reshape(-1, 2)
    depth_vec = img_data[coords_uv]
    # filter out the zero depth region to avoid numerical error
    depth_vec[depth_vec == 0] = 1e-3
    coords_xy = coords_uv
    ext_c = np.ones((coords_xy.shape[0], 1))
    coords_xy = np.hstack((coords_xy, ext_c))
    coords_xyc = np.matmul(np.linalg.inv(K), coords_xy)
    coords_xyz = np.hstack((coords_xyc[:, :2], depth_vec))
    return coords_xyz

def plot(C, L):
    import matplotlib.pyplot as plt
    mask = L == 0
    cC = C[mask].t().numpy()
    plt.scatter(cC[0], cC[1], c='r', s=0.1)
    mask = L == 1
    cC = C[mask].t().numpy()
    plt.scatter(cC[0], cC[1], c='b', s=0.1)
    plt.show()

class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_time = 0
        self.calls = 0
        self.start_time = 0
        self.diff = 0
        self.averate_time = 0
        self.min_time = np.Inf

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=False):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if self.diff < self.min_time:
            self.min_time = self.diff
        if average:
            return self.average_time
        else:
            return self.diff


class InfSampler(Sampler):
    """Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        else:
            perm = torch.arange(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)


def seed_all(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


class RandomLineDataset(Dataset):

    # Warning: read using mutable obects for default input arguments in python.
    def __init__(
        self,
        angle_range_rad=[-np.pi, np.pi],
        line_params=[
            -1,  # Start
            1,  # end
        ],
        is_linear_noise=True,
        dataset_size=100,
        num_samples=10000,
        quantization_size=0.005):

        self.angle_range_rad = angle_range_rad
        self.is_linear_noise = is_linear_noise
        self.line_params = line_params
        self.dataset_size = dataset_size
        self.rng = np.random.RandomState(0)

        self.num_samples = num_samples
        self.num_data = int(0.2 * num_samples)
        self.num_noise = num_samples - self.num_data

        self.quantization_size = quantization_size

    def __len__(self):
        return self.dataset_size

    def _uniform_to_angle(self, u):
        return (self.angle_range_rad[1] -
                self.angle_range_rad[0]) * u + self.angle_range_rad[0]

    def _sample_noise(self, num, noise_params):
        noise = noise_params[0] + self.rng.randn(num, 1) * noise_params[1]
        return noise

    def _sample_xs(self, num):
        """Return random numbers between line_params[0], line_params[1]"""
        return (self.line_params[1] - self.line_params[0]) * self.rng.rand(
            num, 1) + self.line_params[0]

    def __getitem__(self, i):
        # Regardless of the input index, return randomized data
        angle, intercept = np.tan(self._uniform_to_angle(
            self.rng.rand())), self.rng.rand()

        # Line as x = cos(theta) * t, y = sin(theta) * t + intercept and random t's
        # Drop some samples
        xs_data = self._sample_xs(self.num_data)
        ys_data = angle * xs_data + intercept + self._sample_noise(
            self.num_data, [0, 0.1])

        noise = 4 * (self.rng.rand(self.num_noise, 2) - 0.5)

        # Concatenate data
        input = np.vstack([np.hstack([xs_data, ys_data]), noise])
        feats = input
        labels = np.vstack(
            [np.ones((self.num_data, 1)),
             np.zeros((self.num_noise, 1))]).astype(np.int32)

        # Quantize the input
        discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
            coords=input,
            feats=feats,
            labels=labels,
            quantization_size=self.quantization_size,
            ignore_label=-100)

        return discrete_coords, unique_feats, unique_labels
