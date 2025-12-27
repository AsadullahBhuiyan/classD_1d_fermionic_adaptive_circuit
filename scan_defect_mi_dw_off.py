import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from classD_1dMFGTN import classD_1d_MFGTN


def _int_env(name, default=None):
    val = os.environ.get(name)
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


# Cap threaded math libraries before heavy imports/compute
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


def average_over_samples(N, nshell, sigma_vals, mu_1, samples=100, n_jobs=None, cycles=20):
    defects = []
    mis = []
    for sigma in tqdm(sigma_vals, desc=f"N={N}, nshell={nshell}"):
        model = classD_1d_MFGTN(N=N, DW=False, nshell=nshell, mu_1=0)
        model.construct_MW_projectors(nshell=nshell)
        res = model.run_adaptive_circuit(
            G_history=True,
            progress=False,
            cycles=cycles,
            postselect=False,
            samples=samples,
            n_jobs=n_jobs,
            backend="loky",
            parallelize_samples=True,
            store="top",
            init_mode="random_pure",
            remember_init=False,
            save=False,
            sigma=sigma,
        )
        top_hist = res["G_hist"]  # shape (samples, cycles, 2N, 2N)
        finals = top_hist[:, -1]
        d_arr = [model.compute_defect_density(Gk) for Gk in finals]
        mi_arr = [model.compute_antipodal_MI(Gk) for Gk in finals]
        defects.append(np.mean(d_arr))
        mis.append(np.mean(mi_arr))
    return np.array(defects), np.array(mis)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=None, help="joblib parallel workers")
    args = parser.parse_args()
    n_jobs = args.n_jobs if args.n_jobs is not None else _CPU_LIMIT

    sigma_vals = np.linspace(0.0, 1.0, 51)
    nshell_list = [1, 2, None]
    N_list = [16, 32, 64]
    samples = 100
    mu = 0 # topological phase

    plt.rcParams.update({"figure.dpi": 120})

    fig, axes = plt.subplots(
        nrows=len(nshell_list),
        ncols=2,
        figsize=(10, 9),
        sharex=True,
        sharey="col",
    )
    fig.subplots_adjust(hspace=0.15, wspace=0.08)
    if len(nshell_list) == 1:
        axes = np.array([axes])

    for row_idx, nshell in enumerate(nshell_list):
        ax_def = axes[row_idx, 0]
        ax_mi = axes[row_idx, 1]

        for N in N_list:
            defects, mis = average_over_samples(N, nshell, sigma_vals, mu_1=mu, samples=samples, n_jobs=n_jobs)
            label = f"N={N}"
            ax_def.plot(sigma_vals, defects, "-o", label=label)
            ax_mi.plot(sigma_vals, mis, "-o", label=label)

            outdir = "results"
            os.makedirs(outdir, exist_ok=True)
            fname = f"defect_mi_dw_off_N{N}_nsh{nshell}_samples{samples}_mu={mu}.npz"
            path = os.path.join(outdir, fname)
            np.savez_compressed(
                path,
                sigma=sigma_vals,
                defects=defects,
                antipodal_mi=mis,
                N=N,
                nshell=nshell,
                samples=samples,
            )

        ax_def.set_ylabel(r"$\overline{\mathrm{defect\ density}}$")
        ax_mi.set_ylabel(r"$\overline{\mathrm{antipodal\ MI}}$")
        ax_def.grid(alpha=0.3)
        ax_mi.grid(alpha=0.3)
        ax_def.set_title(f"nshell={nshell}")
        ax_mi.set_title(f"nshell={nshell}")

    axes[-1, 0].set_xlabel(r"$\sigma$")
    axes[-1, 1].set_xlabel(r"$\sigma$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(N_list), bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs("results", exist_ok=True)
    fig.savefig(os.path.join("results", f"defect_mi_dw_off_N{N}_nsh{nshell}_samples{samples}_mu={mu}.png"), dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
