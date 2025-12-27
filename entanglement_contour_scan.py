#!/usr/bin/env python3
"""
Track entanglement contour history for a maximally mixed top layer.
CPU/thread limits are respected via environment variables.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import importlib
import classD_1dMFGTN  
importlib.reload(classD_1dMFGTN)
from classD_1dMFGTN import classD_1d_MFGTN

def _int_env(name, default=None):
    val = os.environ.get(name)
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


# Cap threaded math libraries before importing NumPy heavy work
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
N = 32
nshell = 1
mu_1, mu_2 = 30, 1
samples = 100

def run_entanglement_contour(cycles=20, samples=samples, progress=True):
    """
    Initialize maximally mixed top layer, run many samples in parallel, and
    return the sample-averaged entanglement contour over time.
    Returns (contour_mean, times, dw_loc).
    """


    model = classD_1d_MFGTN(N=N, DW=True, nshell=nshell, mu_1=mu_1, mu_2=mu_2)
    result = model.run_adaptive_circuit(
        cycles=cycles,
        G_history=True,
        progress=progress,
        postselect=False,
        samples=samples,
        parallelize_samples=True,
        n_jobs=_CPU_LIMIT,
        store="top",
        init_mode="maxmix",
        save=False,
    )

    # G_hist shape: (S, T, 2N, 2N) with T = cycles+1 when remember_init=True
    G_hist = result["G_hist"]
    S, T = G_hist.shape[0], G_hist.shape[1]
    sites = np.arange(N)

    contour_samples = []
    desc = "Computing entanglement contour (samples x time)"
    for s in tqdm(range(S), desc=desc):
        contours_t = []
        for t in range(T):
            contours_t.append(model.compute_entanglement_contour(G_hist[s, t], sites))
        contour_samples.append(np.stack(contours_t, axis=0))

    contour_samples = np.stack(contour_samples, axis=0)  # (S, T, N)
    contour_mean = np.mean(contour_samples, axis=0)      # (T, N)
    times = np.arange(T)
    dw_loc = getattr(model, "DW_loc", (None, None))
    return contour_mean, times, dw_loc


def _majorana_xticks(n_sites, label_every=2):
    """
    Return tick positions/labels matching interleaved Majorana ordering
    [gamma1_0, gamma2_0, gamma1_1, gamma2_1, ...].
    """
    n_majoranas = 2 * n_sites
    ticks = np.arange(n_majoranas)
    labels = [str(i // 2) if (i % 2 == 0 and (i // 2) % label_every == 0) else "" for i in ticks]
    return ticks, labels


def plot_contour_heatmap(contour_mean, samples, times, dw_loc, outdir="results"):
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f"entanglement_contour_maxmix_{samples}_samples.png")
    data_path = os.path.join(outdir, f"entanglement_contour_maxmix_{samples}_samples.npz")

    # Save data
    np.savez_compressed(
        data_path,
        contour_mean=contour_mean,
        times=times,
        dw_loc=np.array(dw_loc),
    )

    # Plot heatmap: horizontal = site index, vertical = time step
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        contour_mean / np.log(2),
        origin="lower",
        aspect="auto",
        cmap="Blues",
        extent=(-0.5, contour_mean.shape[1] - 0.5, -0.5, contour_mean.shape[0] - 0.5),
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$s_i / \log 2$")

    ax.set_xlabel("Interleaved Majorana index")
    ax.set_ylabel("Time step")
    ax.set_title(
        f"Entanglement Contour (maximally mixed ini) | {samples} samples | N={N} | nshell={nshell}"
    )
    # Grid overlay aligned with cell edges and unit-cell tick marks
    # Major ticks every 1 cell (labels every 2 sites to avoid crowding)
    xticks, xlabels = _majorana_xticks(contour_mean.shape[1] // 2, label_every=2)
    ax.set_xticks(xticks)  # left edge of each cell
    ax.set_xticklabels(xlabels)
    ax.set_yticks(np.arange(contour_mean.shape[0]))
    ax.set_yticklabels([str(y) if y % 5 == 0 else "" for y in range(contour_mean.shape[0])])

    # Minor ticks at cell edges for gridlines
    ax.set_xticks(np.arange(-0.5, contour_mean.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, contour_mean.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", alpha=0.4, linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    # Annotate domain wall locations
    if dw_loc is not None and len(dw_loc) == 2:
        dw_text = f"DWs at x = {dw_loc[0]}, {dw_loc[1]}"
        ax.text(
            0.02,
            0.98,
            dw_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )

    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"Saved contour data to {data_path}")
    print(f"Saved contour heatmap to {fname}")


def plot_contour_curves(contour_mean, times, time_points=(0, 5, 10, 20), outdir="results", save_plot=False):
    """
    Plot entanglement contour curves vs interleaved Majorana index at selected times.
    contour_mean: array (T, 2*L) already computed.
    """
    os.makedirs(outdir, exist_ok=True)
    max_t = contour_mean.shape[0] - 1
    sel_times = [t for t in time_points if 0 <= t <= max_t]
    if not sel_times:
        raise ValueError(f"No valid times in {time_points}; available range 0..{max_t}")

    fig, ax = plt.subplots(figsize=(10, 5))
    majorana_idx = np.arange(contour_mean.shape[1])
    for t in sel_times:
        ax.plot(
            majorana_idx,
            contour_mean[t] / np.log(2),
            "-o",
            markersize=3,
            label=f"t={t}",
        )

    xticks, xlabels = _majorana_xticks(contour_mean.shape[1] // 2, label_every=2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Interleaved Majorana index (gamma1_i, gamma2_i)")
    ax.set_ylabel(r"$s_i / \log 2$")
    ax.set_title("Entanglement Contour Slices vs Interleaved Majorana Index")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    if save_plot:
        fname = os.path.join(outdir, "entanglement_contour_curves.png")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"Saved contour curves plot to {fname}")
    return fig, ax


def main():
    t0 = time.time()
    samples = 100
    contour_mean, times, dw_loc = run_entanglement_contour(cycles=20, samples=samples, progress=True)
    plot_contour_heatmap(contour_mean, samples, times, dw_loc)
    plot_contour_curves(contour_mean, times, time_points=(0, 5, 10, 20), save_plot=True)
    elapsed = time.time() - t0
    print(f"Total time elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
