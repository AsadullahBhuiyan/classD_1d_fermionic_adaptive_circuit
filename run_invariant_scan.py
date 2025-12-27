#!/usr/bin/env python3
"""
Scan entanglement zero-mode count vs mu for classD_1d_MFGTN.
CPU/thread limits are respected via environment variables.
"""

import os
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

import importlib
import classD_1dMFGTN  # make sure this points to the lowercase file

importlib.reload(classD_1dMFGTN)
from classD_1dMFGTN import classD_1d_MFGTN

def run_config(mu_val, t_val, N, nshell, samples=100, cycles=20, n_jobs=None):
    """
    Run adaptive circuit for given (mu, N, nshell) and return entanglement zero modes.
    traj_resolved: compute zero modes per trajectory then average.
    traj_avg: average G over trajectories first, then compute zero modes once.
    Uses DW=False and parallelizes over samples.
    """
    model = classD_1d_MFGTN(N=N, DW=False, nshell=nshell, mu_1=mu_val, mu_2=mu_val, t_1=t_val, t_2=t_val)
    result = model.run_adaptive_circuit(
        cycles=cycles,
        G_history=True,
        progress=True,
        postselect=False,
        samples=samples,
        parallelize_samples=True,
        n_jobs=n_jobs,
        store="top",
        init_mode="random_pure",
        save=False,
        multisample_cycle_threshold=0,
    )

    G_hist = result["G_hist"]  # (S, T, 2N, 2N)
    steady = np.asarray(G_hist[:, -1])  # last cycle per sample

    traj_resolved = []
    for Gs in steady:
        traj_resolved.append(model.ee_zero_modes(Gs, N_A=N // 2))
    traj_resolved = np.array(traj_resolved)

    G_avg = np.mean(steady, axis=0)
    G_avg = 0.5 * (G_avg - G_avg.T)  # enforce skew-symmetry
    traj_avg = model.ee_zero_modes(G_avg, N_A=N // 2)
    return traj_resolved, traj_avg


def main():
    t0 = time.time()
    mu_list = np.linspace(1.5, 2.5, 21)
    N_list = [16]
    nshell_list = [1,2,3,None]
    samples = 10
    cycles = 20
    n_jobs = _CPU_LIMIT
    t_val = 1

    traj_resolved_mean = {}
    traj_resolved_std = {}
    traj_avg_vals = {}

    for N in N_list:
        for nshell in nshell_list:
            key = (N, nshell)
            traj_resolved_mean[key] = []
            traj_resolved_std[key] = []
            traj_avg_vals[key] = []
            for mu_val in tqdm(mu_list, desc=f"N={N}, nshell={nshell}"):
                traj_resolved, traj_avg = run_config(mu_val, t_val, N, nshell, samples=samples, cycles=cycles, n_jobs=n_jobs)
                traj_resolved_mean[key].append(np.mean(traj_resolved))
                traj_resolved_std[key].append(np.std(traj_resolved))
                traj_avg_vals[key].append(traj_avg)

    # Save raw data
    os.makedirs("results", exist_ok=True)
    data_path = os.path.join("results", "zero_mode_scan.npz")
    np.savez_compressed(
        data_path,
        mu_list=mu_list,
        N_list=N_list,
        nshell_list=np.array(nshell_list, dtype=object),
        traj_resolved_mean=traj_resolved_mean,
        traj_resolved_std=traj_resolved_std,
        traj_avg_vals=traj_avg_vals,
    )

    # Plot in 4x2 grid: left = traj-resolved, right = traj-averaged
    fig, axes = plt.subplots(
        nrows=len(nshell_list),
        ncols=2,
        figsize=(12, 9),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)
    colors = cm.viridis(np.linspace(0.2, 0.9, len(N_list)))
    markers = ["o", "s", "^", "D", "P"]

    for row_idx, nshell in enumerate(nshell_list):
        ax_res = axes[row_idx, 0]
        ax_avg = axes[row_idx, 1]
        nshell_label = r"\infty" if nshell is None else nshell

        for ci, N in enumerate(N_list):
            key = (N, nshell)
            label = f"N={N}"
            ax_res.plot(
                mu_list,
                traj_resolved_mean[key],
                marker=markers[ci % len(markers)],
                color=colors[ci],
                label=label,
                linewidth=1.2,
                markersize=4,
            )
            ax_avg.plot(
                mu_list,
                traj_avg_vals[key],
                marker=markers[ci % len(markers)],
                color=colors[ci],
                label=label,
                linewidth=1.2,
                markersize=4,
            )

        ax_res.set_title(f"traj-resolved | nshell = {nshell_label}")
        ax_avg.set_title(f"traj-averaged | nshell = {nshell_label}")
        ax_res.grid(True, alpha=0.3)
        ax_avg.grid(True, alpha=0.3)
        ax_res.set_ylabel(r"zero modes")

    axes[-1, 0].set_xlabel(r"$\mu$")
    axes[-1, 1].set_xlabel(r"$\mu$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(N_list), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fig_path = os.path.join("results", f"zero_modes_vs_mu_mu{mu_list[0]:.1f}-{mu_list[-1]:.1f}.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved data to {data_path}")
    print(f"Saved figure to {fig_path}")
    elapsed = time.time() - t0
    print(f"Total time elapsed: {elapsed/3600:.2f} h ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
