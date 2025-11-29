#!/usr/bin/env python3
"""
Scan antipodal mutual information vs mu for classD_1d_MFGTN.
CPU/thread limits are respected via environment variables.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm


def _int_env(name, default=None):
    val = os.environ.get(name)
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


# Cap threaded math libraries before importing NumPy/SciPy heavy work
_CPU_LIMIT = (
    _int_env("SLURM_CPUS_PER_TASK")
    or _int_env("MY_CPU_COUNT")
    or (os.cpu_count() or 1)
)
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
    os.environ[var] = str(_CPU_LIMIT)
os.environ.setdefault("MY_CPU_COUNT", str(_CPU_LIMIT))

try:
    os.sched_setaffinity(0, set(range(_CPU_LIMIT)))
    print(f"[info] CPU affinity pinned to {_CPU_LIMIT} cores.")
except Exception as e:
    print(f"[warn] Could not set CPU affinity: {e}")

from classD_1dMFGTN import classD_1d_MFGTN


def run_config(mu_val, N, nshell, samples=100, cycles=20, n_jobs=None):
    """
    Run adaptive circuit for given (mu, N, nshell) and return MI per sample.
    Uses DW=False and parallelizes over samples.
    """
    model = classD_1d_MFGTN(N=N, DW=False, nshell=nshell, mu_1=mu_val, mu_2=mu_val)
    result = model.run_adaptive_circuit(
        cycles=cycles,
        G_history=True,
        progress=False,
        postselect=False,
        samples=samples,
        parallelize_samples=True,
        n_jobs=n_jobs,
        store="top",
        init_mode="random_pure",
        save=False,
    )

    G_hist = result["G_hist"]  # (S, T, 2N, 2N)
    steady = G_hist[:, -1]     # last cycle per sample

    mi_vals = [model.compute_antipodal_MI(Gs) for Gs in steady]
    return np.array(mi_vals)


def main():
    t0 = time.time()
    mu_list = np.unique(np.concatenate((np.linspace(1.0, 1.8, 21), np.linspace(1.8, 2.2 , 51), np.linspace(2.2, 3.0, 21))))
    N_list = [32, 64, 128]
    nshell_list = [1, 2, 3, None]
    samples = 100
    cycles = 20
    n_jobs = _CPU_LIMIT

    mi_mean = {}
    mi_std = {}

    for N in N_list:
        for nshell in nshell_list:
            key = (N, nshell)
            mi_mean[key] = []
            mi_std[key] = []
            for mu_val in tqdm(mu_list, desc=f"N={N}, nshell={nshell}"):
                mi_vals = run_config(mu_val, N, nshell, samples=samples, cycles=cycles, n_jobs=n_jobs)
                mi_mean[key].append(np.mean(mi_vals))
                mi_std[key].append(np.std(mi_vals))

    # Save raw data
    os.makedirs("results", exist_ok=True)
    data_path = os.path.join("results", "mi_scan_finer_near_mu=2_bigger_system_size.npz")
    np.savez_compressed(
        data_path,
        mu_list=mu_list,
        N_list=N_list,
        nshell_list=np.array(nshell_list, dtype=object),
        mi_mean=mi_mean,
        mi_std=mi_std,
    )

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = cm.viridis(np.linspace(0.2, 0.9, len(N_list)))
    markers = ["o", "s", "^", "D", "P"]

    for ci, N in enumerate(N_list):
        for mi, nshell in enumerate(nshell_list):
            key = (N, nshell)
            nshell_label = r"\infty" if nshell is None else nshell
            label = rf"$N={N},\ \mathrm{{nshell}}={nshell_label}$"
            ax.plot(
                mu_list,
                mi_mean[key],
                marker=markers[mi % len(markers)],
                color=colors[ci],
                label=label,
                linewidth=1.2,
                markersize=4,
            )

    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\overline{I(A:B)}$")
    ax.set_title(r"Antipodal Mutual Information vs. $\mu$")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2, fontsize=10)
    fig.tight_layout()

    # informative filename: include mu range
    fig_path = os.path.join("results", f"mi_vs_mu_mu{mu_list[0]:.1f}-{mu_list[-1]:.1f}.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved data to {data_path}")
    print(f"Saved figure to {fig_path}")
    elapsed = time.time() - t0
    print(f"Total time elapsed: {elapsed/3600:.2f} h ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
