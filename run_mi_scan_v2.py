#!/usr/bin/env python3
"""
Scan antipodal mutual information vs mu for classD_1d_MFGTN with periodic Gaussian envelopes.
System size fixed at N=32, cycles=32, multi-sample threshold = cycles//2 (default).
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


# Cap threaded math libraries before importing heavy work
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

import importlib
import classD_1dMFGTN  # make sure this points to the lowercase file

importlib.reload(classD_1dMFGTN)
from classD_1dMFGTN import classD_1d_MFGTN


def run_config(mu_val, N, samples=100, cycles=None, n_jobs=None):
    """
    Run adaptive circuit for given (mu, N) and return MI per sample.
    Uses DW=False and parallelizes over samples.
    """
    cycles = N if cycles is None else int(cycles)
    threshold_cycles = 0  # 3N/4
    model = classD_1d_MFGTN(
        N=N,
        DW=False,
        nshell=1,
        envelope_width=None,
        mu_1=mu_val,
        mu_2=mu_val,
    )
    result = model.run_adaptive_circuit(
        cycles=cycles,
        G_history=False,
        progress=True,
        postselect=False,
        samples=samples,
        parallelize_samples=True,
        n_jobs=n_jobs,
        store="top",
        init_mode="random_pure",
        save=True,
        multisample_cycle_threshold=threshold_cycles,
        pre_threshold_progress=True,
        clear_output_each_cycle=True
    )

    # Single unified key from run_adaptive_circuit: always (S, 2N, 2N)
    if "steady_state" in result:
        steady_states = result["steady_state"]
    elif "G_hist" in result and result["G_hist"] is not None:
        steady_states = result["G_hist"][:, -1]  # fallback to last cycle if histories kept
    else:
        raise KeyError("run_adaptive_circuit result missing steady_state data.")

    mi_vals = [model.compute_antipodal_MI(Gs) for Gs in steady_states]
    return np.array(mi_vals)


def main():
    t0 = time.time()
    mu_list = np.linspace(1, 2.2, 31)
    samples = 10
    n_jobs = _CPU_LIMIT
    N_list = [16, 32, 64, 128]

    mi_mean = {}
    mi_std = {}

    for N in N_list:
        key = N
        mi_mean[key] = []
        mi_std[key] = []
        for mu_val in tqdm(mu_list, desc=f"N={N}"):
            mi_vals = run_config(mu_val, N, samples=samples, cycles=N, n_jobs=n_jobs)
            mi_mean[key].append(np.mean(mi_vals))
            mi_std[key].append(np.std(mi_vals))

    # Save raw data
    os.makedirs("results", exist_ok=True)
    data_path = os.path.join("results", "mi_scan_vs_mu_vs_N_nshell=1.npz")
    np.savez_compressed(
        data_path,
        mu_list=mu_list,
        N_list=np.array(N_list, dtype=int),
        mi_mean=mi_mean,
        mi_std=mi_std,
        cycles_per_N={N: N for N in N_list},
        samples=samples,
    )

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = cm.viridis(np.linspace(0.15, 0.9, len(N_list)))
    marker = "o"

    for ci, N in enumerate(N_list):
        key = N
        label = rf"$N={N}$"
        mean_arr = np.array(mi_mean[key])
        std_arr = np.array(mi_std[key])
        ax.errorbar(
            mu_list,
            mean_arr,
            yerr=std_arr/np.sqrt(samples),
            marker=marker,
            color=colors[ci],
            label=label,
            linewidth=1.0,
            markersize=4,
            elinewidth=0.8,
            capsize=2.5,
            alpha=0.9,
        )

    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\overline{I(A:B)}$")
    ax.set_title(rf"Antipodal Mutual Information vs. $\mu$, nshell = 1, samples={samples}")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()

    fig_path = os.path.join("results", f"mi_vs_mu_vs_N_mu{mu_list[0]:.1f}-{mu_list[-1]:.1f}_nshell=1.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved data to {data_path}")
    print(f"Saved figure to {fig_path}")
    elapsed = time.time() - t0
    print(f"Total time elapsed: {elapsed/3600:.2f} h ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
