import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import importlib
import classD_1dMFGTN  # make sure this points to the lowercase file
importlib.reload(classD_1dMFGTN)
from classD_1dMFGTN import classD_1d_MFGTN
# Thread caps
cpu_limit = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("MY_CPU_COUNT") or (os.cpu_count() or 1))
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
    os.environ[var] = str(cpu_limit)

# Params
mu_list = np.linspace(1, 2.2, 31)
N_list = [16, 32, 64, 128]
samples = 10
cycles_per_N = 1  # cycles = cycles_per_N * N

corr_mean = {N: [] for N in N_list}
corr_std = {N: [] for N in N_list}

for N in N_list:
    cycles = int(cycles_per_N * N)
    for mu_val in mu_list:
        model = classD_1d_MFGTN(
            N=N,
            DW=False,
            nshell=1,
            envelope_width=None,
            mu_1=mu_val,
            mu_2=mu_val,
        )
        out = model.run_adaptive_circuit(
            cycles=cycles,
            samples=samples,
            parallelize_samples=True,
            n_jobs=cpu_limit,
            store="top",
            init_mode="random_pure",
            progress=False,
            save=True,
            multisample_cycle_threshold=0,
            clear_output_each_cycle=True,
            save_suffix="v2",
        )
        if "steady_state" in out:
            Gs = out["steady_state"]
        elif "G_hist" in out and out["G_hist"] is not None:
            Gs = out["G_hist"][:, -1]
        else:
            raise KeyError("run_adaptive_circuit result missing steady_state")

        C_stack = np.stack([model.square_correlation_function(G) for G in Gs], axis=0)
        corr_mean[N].append(C_stack.mean(axis=0))
        corr_std[N].append(C_stack.std(axis=0))

# Plotting
fig, axes = plt.subplots(1, len(N_list), figsize=(5 * len(N_list), 4), sharey=True)
axes = np.atleast_1d(axes)
colors = cm.plasma(np.linspace(0.1, 0.95, len(mu_list)))

for ai, N in enumerate(N_list):
    ax = axes[ai]
    for ci, mu_val in enumerate(mu_list):
        C_mean = np.array(corr_mean[N][ci])
        X_vals = np.arange(C_mean.size)
        ax.plot(
            X_vals,
            C_mean,
            color=colors[ci],
            label=rf"$\mu={mu_val:.2f}$",
            linewidth=1.0,
            alpha=0.9,
        )
    ax.set_xlabel(r"$X$")
    ax.set_title(rf"$N={N}$, cycles={int(cycles_per_N * N)}, samples={samples}$")
    ax.grid(True, alpha=0.3)
    if ai == 0:
        ax.set_ylabel(r"$C(X)$")
    ax.legend(frameon=False, fontsize=7, ncol=1)

fig.suptitle(r"Square correlation $C(X)$ vs $X$ for varying $\mu$ (nshell = 1)", y=1.02)
fig.tight_layout()
plt.savefig("results/correlation_function_vs_mu_nshell1_v2.png", dpi=300, bbox_inches="tight")
plt.show()
