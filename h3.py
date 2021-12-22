#! /usr/bin/env python3
"""Homework 3.

usage: h3.py [-hVnS]

options:
    -h, --help                  Show this screen.
    -V, --version               Show version.
    -n, --dry-run               Don't open display window.
    -S, --save                  Save the figure.
"""
if __name__ == "__main__" and __doc__:
    from docopt import docopt
    from typing import Dict

    try:
        args: Dict[str, str] = docopt(__doc__, version="v0.0.1")
    except Exception:
        args = {}

    import numpy as np
    from scipy.linalg import toeplitz
    import padasip as pa
    from padasip.filters import FilterLMS, FilterRLS
    from timeit import timeit
    from typing import Final
    from matplotlib import pyplot as plt

    sigma: Final = np.sqrt(0.01)
    h: Final = np.r_[0.6, 0.7]
    H: Final = toeplitz(np.r_[h, np.zeros(5)], np.zeros(6)).T
    R: Final = H @ np.eye(7) @ H.T + sigma ** 2 * np.eye(6)
    f_MMSE: Final = np.linalg.inv(R) @ H[:, 0]
    eigs, _ = np.linalg.eigh(R)

    T: Final = 50
    N: Final = 500
    L: Final = len(h)
    hs = np.zeros([2, T, N, L])
    es = np.zeros(hs.shape[:-1])
    ts = np.zeros(hs.shape[:2])
    for t in range(T):
        xs = np.random.choice([1, -1], N + L - 1)
        X = pa.input_from_history(xs, L)
        u = X @ h
        u += sigma ** 2 * np.random.randn(*u.shape)
        lms = FilterLMS(n=L, w="zeros", mu=0.1 / max(eigs))
        rls = FilterRLS(n=L, w="zeros", mu=0.005)
        _, es[0, t], hs[0, t] = lms.run(u, X)
        _, es[1, t], hs[1, t] = rls.run(u, X)
    es1 = np.mean(es, 1)
    hs1 = np.mean(hs, 1)

    name = "estimation_errors"
    fig, ax = plt.subplots(num=name)
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$e$")
    ax.plot(es1[0], label="lms")
    ax.plot(es1[1], label="rls")
    plt.legend()
    if args.get("--save"):
        import os

        dir = "images"
        os.makedirs(dir, exist_ok=True)
        fig.savefig(os.path.join(dir, f"{name}.png"))
    if not args.get("--dry-run"):
        plt.show()
        name = "filters"

    for t in range(T):
        xs = np.random.choice([1, -1], N + L - 1)
        X = pa.input_from_history(xs, L)
        u = X @ h
        u += sigma ** 2 * np.random.randn(*u.shape)
        lms = FilterLMS(n=L, w="zeros", mu=0.1 / max(eigs))
        rls = FilterRLS(n=L, w="zeros", mu=0.005)
        ts[0, t] = timeit("lms.run(u, X)", number=1, globals=globals())
        ts[1, t] = timeit("rls.run(u, X)", number=1, globals=globals())

    name = "estimation_time"
    fig, ax = plt.subplots(num=name)
    ax.set_xlabel(r"$t$/s")
    ax.set_ylabel(r"$n$")
    ax.hist(ts[0], label="lms")
    ax.hist(ts[1], label="rls")
    plt.legend()
    if args.get("--save"):
        import os

        dir = "images"
        os.makedirs(dir, exist_ok=True)
        fig.savefig(os.path.join(dir, f"{name}.png"))
    if not args.get("--dry-run"):
        plt.show()
