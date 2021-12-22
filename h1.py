#! /usr/bin/env python3
"""Homework 1.

usage: h1.py [-hV]

options:
    -h, --help                  Show this screen.
    -V, --version               Show version.
"""
if __name__ == "__main__" and __doc__:
    from docopt import docopt
    from typing import Dict, Union, List

    Arg = Union[bool, int, str, List[str]]
    args: Dict[str, Arg] = docopt(__doc__, version="v0.0.1")

    import padasip as pa
    import numpy as np
    import math
    import os
    from matplotlib import pyplot as plt
    from scipy.linalg import toeplitz
    from scipy.signal import correlate

    N = 5000
    L = 10
    sigma = math.sqrt(0.0121)
    hs = np.zeros([2, 2, 4])
    hs[0, 0] = np.array([0.6, -0.2, 0, 0.8])
    hs[0, 1] = np.array([-0.7, 0.4, -0.5, 0])
    hs[1, 0] = np.array([0, 0, 0.38, -0.16])
    hs[1, 1] = np.array([0.1, 0.2, 0.2, 0.9])
    H = hs.reshape(2, 8).T
    k = len(hs[0, 0])
    X1 = np.zeros([])
    X = np.zeros([])
    Y = np.zeros([])
    times = 100
    hs_hat = np.zeros([2, 2, times, k])
    for i in range(times):
        x1 = np.random.choice([1, -1], N + k - 1)
        X1 = pa.input_from_history(x1, n=k)
        x2 = np.random.choice([1, -1], N + k - 1)
        X2 = pa.input_from_history(x2, n=k)
        X = np.c_[X1, X2]
        Y = X @ H
        W = np.random.randn(*Y.shape) * sigma
        Y += W
        H_hat = np.linalg.lstsq(X, Y, rcond=None)[0]
        # split H_hat to hs_hat[:, :]
        hs_hat[:, :, i] = np.squeeze(
            np.stack(np.split(np.stack(np.split(H_hat, 2)), 2, -1), 1)
        )
    hs_hat_1 = hs_hat[:, :, 0, :]
    hs_hat_100 = hs_hat.mean(2)

    K = int(N / L)
    hs_hat2 = np.zeros([2, 2, K, k])
    errors = np.zeros([2, 2, K])
    for i in range(K):
        X_sample = X[: (i + 1) * L]
        Y_sample = Y[: (i + 1) * L]
        H_hat = np.linalg.lstsq(X_sample, Y_sample, rcond=None)[0]
        hs_hat2[:, :, i] = np.squeeze(
            np.stack(np.split(np.stack(np.split(H_hat, 2)), 2, -1), 1)
        )
        errors[:, :, i] = np.linalg.norm(hs_hat2[:, :, i] - hs, 2, -1)
    name = "errors"
    fig, ax = plt.subplots(num=name)
    ylabel = r"$||\hat{\mathbf{h}}_{ij} - \mathbf{h}_{ij}||$"
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(ylabel)
    for i in range(errors.shape[0]):
        for j in range(errors.shape[1]):
            ij = str(i + 1) + str(j + 1)
            ax.plot(errors[i, j], label=ylabel.replace("ij", ij))
    plt.legend()
    dir = "images"
    os.makedirs(dir, exist_ok=True)
    plt.savefig(os.path.join(dir, f"{name}.png"))

    sinrs = np.zeros(len(Y))
    for i in range(len(Y)):
        h_sample = hs_hat2[0, :, int(np.floor(i / 10))]
        x1_sample = X1[i]
        y_sample = Y[i]
        z = y_sample - h_sample @ x1_sample
        R = toeplitz(correlate(z, z)[-len(z):])
        sinrs[i] = np.mean(np.var(x1_sample) * h_sample.T @ np.linalg.inv(R) @ h_sample)
    sinrs_mean = sinrs.reshape(K, -1).mean(-1).cumsum(-1)
    for i in range(K):
        sinrs_mean[i] /= i + 1
    name = "sinr"
    fig, ax = plt.subplots(num=name)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"SINR")
    plt.savefig(os.path.join(dir, f"{name}.png"))
