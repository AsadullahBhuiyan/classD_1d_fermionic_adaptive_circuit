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


def average_contour_over_samples(N, nshell, sigma_vals, samples=100, n_jobs=None, cycles=20):
    sites_A = np.arange(N // 2)
    contour_len = 2 * len(sites_A)  # majorana_contour=True
    contours = np.zeros((len(sigma_vals), contour_len))
    for i, sigma in enumerate(tqdm(sigma_vals, desc=f"N={N}, nshell={nshell}")):
        model = classD_1d_MFGTN(N=N, DW=True, nshell=nshell)
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
        cont_arr = [
            model.compute_entanglement_contour(
                Gk, sites_A=sites_A, majorana_contour=True
            )
            for Gk in finals
        ]
        contours[i] = np.mean(cont_arr, axis=0)
    return contours, np.arange(contour_len)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=None, help="joblib parallel workers")
    args = parser.parse_args()
    n_jobs = args.n_jobs if args.n_jobs is not None else _CPU_LIMIT

    sigma_vals = np.linspace(0.0, 1.0, 51)
    nshell_list = [None]
    N_list = [32]
    samples = 100

    plt.rcParams.update({"figure.dpi": 120})

    for nshell in nshell_list:
        for N in N_list:
            contours, sites = average_contour_over_samples(
                N, nshell, sigma_vals, samples=samples, n_jobs=n_jobs
            )

            contour_mean = contours / np.log(2)
            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(
                contour_mean,
                origin="lower",
                aspect="auto",
                cmap="Blues",
                extent=(-0.5, contour_mean.shape[1] - 0.5, sigma_vals[0], sigma_vals[-1]),
            )
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(r"$\overline{s_i}/\log 2$")

            ax.set_xlabel("Local Majorana index")
            ax.set_ylabel(r"$\sigma$")
            ax.set_title(
                rf"Half-System Entanglement Contour $(\log 2)$ | {samples} samples | sites={N} | nshell={nshell}"
            )

            ax.set_xticks(np.arange(contour_mean.shape[1]))
            ax.set_xticklabels([str(x) if x % 5 == 0 else "" for x in range(contour_mean.shape[1])])
            ax.set_yticks(sigma_vals)
            ax.set_yticklabels([f"{s:.2f}" if idx % 5 == 0 else "" for idx, s in enumerate(sigma_vals)])

            ax.set_xticks(np.arange(-0.5, contour_mean.shape[1], 1)[0::2], minor=True)
            ax.set_yticks(sigma_vals, minor=True)
            ax.grid(which="minor", color="black", alpha=0.4, linewidth=0.5)
            ax.tick_params(which="minor", length=0)

            dw_loc = getattr(classD_1d_MFGTN(N=N, DW=True, nshell=nshell), "DW_loc", None)
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

            outdir = "results"
            os.makedirs(outdir, exist_ok=True)
            fname = f"ss_contour_dw_on_vs_error_N{N}_nsh{nshell}_samples{samples}.npz"
            path = os.path.join(outdir, fname)
            np.savez_compressed(
                path,
                sigma=sigma_vals,
                contour=contours,
                majorana_indices=sites,
                N=N,
                nshell=nshell,
                samples=samples,
            )

            fname_plot = f"ss_contour_dw_on_vs_error_N{N}_nsh{nshell}_samples{samples}.png"
            fig.savefig(os.path.join(outdir, fname_plot), dpi=300, bbox_inches="tight")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
