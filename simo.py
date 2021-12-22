#! /usr/bin/env python3
"""A simulation program of SIMO.

usage: simo.py [-hVnS] [-s <seed>] [-t <times>] [-k <vectors>] [-l <length>]
    [-a <name>] [-v <var>] [-N <noise>]

options:
    -h, --help              Show this screen.
    -V, --version           Show version.
    -n, --dry-run           Don't open display window.
    -S, --save              Save the figure.
    -s, --seed <seed>       Random seed. [default: 42]
    -t, --times <times>     Times of Monte Carlo simulations. [default: 100]
    -k, --vector <vector>   Number of vectors. [default: 500]
    -l, --length <length>   Length of a vector. [default: 10]
    -a, --name <name>       Name of figure. [default: simo]
    -v, --var <var>         Variance of noise. [default: 0.0121]
    -N, --noise <noise>     Number of noises. 0 means auto. [default: 0]
"""
if __name__ == "__main__" and __doc__:
    from docopt import docopt
    from typing import Dict

    try:
        args: Dict[str, str] = docopt(__doc__, version="v0.0.1")
    except Exception:
        args = {}

    from typing import Final
    import numpy as np
    import math
    import os
    from matplotlib import pyplot as plt
    from scipy.linalg import toeplitz
    from scipy.signal import lfilter
    from tqdm import trange
    import pytorch_lightning as pl

    seed = int(args.get("--seed", "42"))
    pl.seed_everything(seed)

    T: Final = int(args.get("--times", "100"))
    K: Final = int(args.get("--vector", "500"))
    L: Final = int(args.get("--length", "10"))
    N: Final = K * L
    # only first transmitter work
    i: Final = 0
    V = np.zeros([])
    errors0 = np.zeros([T, K])
    sigma: Final = math.sqrt(float(args.get("--var", "0.0121")))
    hs = np.zeros([2, 2, 4])
    hs[0, 0] = np.array([0.6, -0.2, 0, 0.8])
    hs[0, 1] = np.array([-0.7, 0.4, -0.5, 0])
    hs[1, 0] = np.array([0, 0, 0.38, -0.16])
    hs[1, 1] = np.array([0.1, 0.2, 0.2, 0.9])
    h = np.r_[hs[0, 0], hs[0, 1]]
    # transmitters x times x input vectors x coefficients
    # i x t x k x 8
    h_hats = np.zeros([2, T, K, len(h)])
    for t in trange(T, desc="Monte Carlo simulation"):
        x1 = np.random.choice([1, -1], N)
        y1 = np.flip(lfilter(hs[0, 0], 1, x1))
        y2 = np.flip(lfilter(hs[0, 1], 1, x1))
        y1 += sigma * np.random.randn(*y1.shape)
        y2 += sigma * np.random.randn(*y2.shape)
        # make y1, y2 alternately
        y0 = np.transpose(np.stack([y1, y2])).reshape(-1)  # type: ignore
        Rs = np.zeros([K, 2 * L, 2 * L])
        for k in range(K):
            l = 2 * (k + 1) * L
            y = y0[l - 2 * L : l]
            Rs[k] = np.outer(y, y)

        for k in trange(K, leave=False, desc="Use k input vectors"):
            R = np.mean(Rs[: k + 1], 0)
            # sort increasingly
            eigs, U = np.linalg.eigh(R)
            num_noise = int(args.get("--noise", 0))
            # auto
            if num_noise == 0:
                # at least one noise
                num_noise = max(np.count_nonzero(eigs <= sigma), 1)
            Uijs = np.zeros([num_noise, *hs.shape[1:], L])
            for n in range(num_noise):
                # because y1, y2 occur alternately
                ujs = [U[::2, n], U[1::2, n]]
                for j in range(hs.shape[1]):
                    Uijs[n, j] = toeplitz(ujs[j], np.zeros(hs.shape[-1])).T
            V = Uijs.transpose(1, 2, 0, 3).reshape(
                [hs.shape[1] * hs.shape[-1], -1]
            )
            W = V @ np.conj(V.T)
            _, U_W = np.linalg.eigh(W)
            u = U_W[:, 0]
            # phase correlation
            u *= np.sign(h * u)
            # gain correlation
            u *= np.linalg.norm(h)
            h_hats[i, t, k] = u
            errors0[t, k] = (
                (sigma * np.linalg.norm(np.linalg.pinv(V))) ** 2 / (k + 1) / L
            )
    h_hat = np.mean(h_hats[i], 0)
    errors = np.linalg.norm(h_hat - h, 2, -1) / np.linalg.norm(h)
    errors1 = np.mean(errors0, 0)

    name = args.get("--name", "simo")
    fig, ax = plt.subplots(num=name)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$||\Delta\mathbf{h}||$")
    ks = np.arange(1, K + 1)
    ax.plot(ks, errors, label="experiment")
    ax.plot(ks, errors1, label="theory")
    plt.legend()
    dir = "images"
    os.makedirs(dir, exist_ok=True)
    if args.get("--save"):
        fig.savefig(os.path.join(dir, f"{name}.png"))
    if not args.get("--dry-run"):
        plt.show()
