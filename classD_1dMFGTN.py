import numpy as np
import os
import time
import sys
from tqdm import tqdm
from tqdm import notebook as tqdm_nb
from joblib import Parallel, delayed, parallel_backend
from threadpoolctl import threadpool_limits
import joblib
from contextlib import contextmanager, nullcontext

# -------------------------- Utilities --------------------------

class _TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
    """Update tqdm whenever a joblib batch finishes."""
    def __init__(self, tqdm_object, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tqdm_object = tqdm_object
    def __call__(self, *args, **kwargs):
        self.tqdm_object.update(n=self.batch_size)
        return super().__call__(*args, **kwargs)

@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager linking joblib's callback to a tqdm progress bar."""
    original_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = lambda *args, **kwargs: _TqdmBatchCompletionCallback(tqdm_object, *args, **kwargs)
    try:
        with tqdm_object as pbar:
            yield pbar
    finally:
        joblib.parallel.BatchCompletionCallBack = original_callback

# -------------------------- Main Class --------------------------

class classD_1d_MFGTN:

    def __init__(
        self,
        N,
        DW=True,
        nshell=None,
        G0=None,
        mu_1=3,
        mu_2=1,
    ):
        '''Initialize lattice dimensions, domain-wall option, and starting covariance.'''
        self.time_init = time.time()
        self.Nx = int(N)
        self.Nlayer_dofs = 2 * int(N)
        self.Ntot_dofs = 4 * int(N)
        self.nshell = nshell
        self.DW = bool(DW)
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        
        # Initialize Domain Wall profile
        if self.DW:
            self.create_domain_wall(mu_1=self.mu_1, mu_2=self.mu_2)

        self.G0 = None if G0 is None else np.array(G0, dtype=np.complex128, copy=True)
        self.G = None
        self.G_history_samples = None
        self.J = np.array([[0, 1], [-1, 0]], dtype=float) # 2x2 symplectic form

        print("------------------------- classD_1d_MFGTN Initialized -------------------------")

    # ----------------------------- Initialization ----------------------------------

    def random_orthogonal(self, N):
        """Generate a Haar-random 2N x 2N orthogonal matrix via QR."""
        rng = np.random.default_rng()
        X = rng.standard_normal((2 * N, 2 * N))
        Q, R = np.linalg.qr(X)
        d = np.sign(np.diag(R))
        Q = Q @ np.diag(d)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1.0
        return Q
    
    def random_pure_majorana_covariance(self, N, random_O=True):
        """Build a random 2N x 2N Majorana-basis covariance matrix."""
        rng = np.random.default_rng()
        diag_signs = np.diag(rng.choice([-1, 1], size=N))
        base = np.kron(self.J, diag_signs)        
        if random_O:
            O = self.random_orthogonal(N)
            cov = O.T @ base @ O
            return cov
        return base
    
    def init_random_pure_top_layer(self):
        """Initialize the top layer as random pure state and bottom layer as random product state."""
        N = self.Nx
        physical_block = self.random_pure_majorana_covariance(N, random_O=True)
        ancilla_block = self.random_pure_majorana_covariance(N, random_O=False)
        cov = self.block_diag(physical_block, ancilla_block)
        return cov 
    
    def init_max_mix_top_layer(self):
        """Initialize the top layer as maximally mixed and bottom layer as random."""
        N = self.Nx
        ancilla_block = self.random_pure_majorana_covariance(N, random_O=False)
        physical_block = np.zeros_like(ancilla_block, dtype=np.complex128)
        cov = self.block_diag(physical_block, ancilla_block)
        return cov 

    # ------------------------------ Utilities ------------------------------

    def _joblib_tqdm_ctx(self, total, desc):
        """Single outer tqdm bar for joblib.Parallel."""
        bar = (tqdm_nb.tqdm if not sys.stdout.isatty() else tqdm)
        return tqdm_joblib(bar(total=total, desc=desc, unit="task")) if total and total > 1 else nullcontext()
    
    def _ensure_outdir(self, path):
        os.makedirs(path, exist_ok=True)
        return path
    
    def block_diag(self, A, B):
        """Minimal block_diag(A,B) without scipy. Returns [[A,0],[0,B]]."""
        n, m = A.shape[0], B.shape[0]
        Z1 = np.zeros((n, m), dtype=A.dtype)
        Z2 = np.zeros((m, n), dtype=B.dtype)
        return np.block([[A, Z1], [Z2, B]])

    def solve_reg(self, K, B, eps=1e-9):
        """
        Solve K X = B with a small Tikhonov ridge if needed; fall back to pinv.
        """
        try:
            return np.linalg.solve(K, B)
        except np.linalg.LinAlgError:
            pass
        n = K.shape[0]
        K_reg = K + eps * np.eye(n, dtype=K.dtype)
        try:
            return np.linalg.solve(K_reg, B)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(K_reg) @ B

    def get_top_layer(self, G):
        """Return the top-layer block of a full covariance."""
        G = np.asarray(G)
        expect = 2 * self.Nx
        if G.shape[0] == expect:
            return G
        if G.shape[0] >= 2 * expect:
            return G[:expect, :expect]
        raise ValueError(f"Covariance too small for top layer: got {G.shape}, expected at least {(2*expect, 2*expect)}")

    def get_bottom_layer(self, G):
        """Return the bottom-layer block of a full covariance."""
        G = np.asarray(G)
        expect = 2 * self.Nx
        if G.shape[0] < 2 * expect:
            raise ValueError(f"Covariance too small for bottom layer: got {G.shape}")
        return G[expect:, expect:]
    
    def _stabilize_covariance(self, G, clip_val=1.0):
        """Enforce antisymmetry and optionally clip entries to [-clip_val, clip_val]."""
        G = 0.5 * (G - G.T)
        if clip_val is not None:
            G = np.clip(np.real(G), -clip_val, clip_val) 
        return G

    # ------------------ Majorana Wannier (MW) Projectors ------------------

    def create_domain_wall(self, mu_1, mu_2):
        '''Construct and store the default domain-wall mass profile.'''
        N = self.Nx
        mu_profile = np.full(N, float(mu_1))
        half = N // 2
        w = max(1, int(np.floor(0.2 * N)))
        x0 = max(0, half - w)
        x1 = min(N, half + w + 1)
        mu_profile[x0:x1] = float(mu_2)
        self.DW_loc = [int(x0-1), int(x1-1)]
        self.mu_profile = mu_profile

    def construct_MW_projectors(self, nshell=None):
        """
        Build 1D Majorana Wannier spinors efficiently using FFT.
        Uses np.vstack to ensure shape (2N, N) per species.
        """
        N = self.Nx
        if self.DW:
            mu_vec = self.mu_profile
        else:
            mu_vec = np.full(N, self.mu_1, float)

        k = 2 * np.pi * np.fft.fftfreq(N, d=1.0)              
        Rgrid = np.arange(N, dtype=float)                     

        dy = -2.0 * np.sin(k)                                 
        dz = -(2.0 * np.cos(k)[:, None] + mu_vec[None, :])    
        dyR = dy[:, None]                                     
        dmag = np.sqrt(dyR * dyR + dz * dz)
        dmag = np.where(dmag == 0, 1e-15, dmag)

        pauli_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        pauli_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        I2 = np.eye(2, dtype=np.complex128)
        tau = (1.0 / np.sqrt(2.0)) * np.array([1, 1], dtype=np.complex128)
        M = np.array([[1, 1], [1, -1]], dtype=np.complex128)  

        dk = (dyR[..., None, None] * pauli_y + dz[..., None, None] * pauli_z) / dmag[..., None, None]
        Pminus = 0.5 * (I2 - dk)                                                                    

        psi_k = np.einsum("m,krmn->krn", tau.conj(), Pminus, optimize=True)                          
        phase = np.exp(-1j * k[:, None] * Rgrid[None, :])                                           
        F = phase[..., None] * psi_k                                                                
        chi = np.fft.ifft(F, axis=0)                                                                

        Pi = np.real(np.einsum("krm,mn->krn", chi, M, optimize=True))                               

        if nshell is not None:
            idx = np.arange(N)
            dist = ((idx[:, None] - Rgrid[None, :] + N // 2) % N) - N // 2
            mask = (np.abs(dist) <= int(nshell)).astype(float)
            Pi *= mask[..., None]

        norms = np.linalg.norm(Pi, axis=0, keepdims=True) + 1e-15                                   
        Pi /= norms
        Pi = np.real_if_close(Pi)

        # Ensure shapes are (2N, N) for the Top layer logic
        # Top block = component 0, Bottom block = 0
        self.MW1 = np.vstack((Pi[:, :, 0].astype(np.float64, copy=True),
                              np.zeros_like(Pi[:, :, 0], dtype=np.float64)))  # (2N, R)
        # Top block = 0, Bottom block = component 1
        self.MW2 = np.vstack((np.zeros_like(Pi[:, :, 1], dtype=np.float64),
                              Pi[:, :, 1].astype(np.float64, copy=True)))      # (2N, R)
  

    # ------------------------------ Real-space Kitaev H and covariance ------------------------------

    def kitaev_majorana_hamiltonian(self, N=None, mu=3.0, DW=False, PBCs=True):
        """
        Build the real-space Majorana Hamiltonian H for the Kitaev chain with t=Δ=1.
        Ordering: [γ1_0..γ1_{N-1}, γ2_0..γ2_{N-1}].
        """
        N = self.Nx if N is None else int(N)
        mu = float(mu)
        if DW:
            if not hasattr(self, "mu_profile"):
                self.create_domain_wall(mu_1=self.mu_1, mu_2=self.mu_2)
            mu_diag = np.diag(self.mu_profile.astype(np.complex128))
            M = -mu_diag - 2 * np.eye(N, k=-1, dtype=np.complex128)
        else:
            M = -mu * np.eye(N, dtype=np.complex128) - 2 * np.eye(N, k=-1, dtype=np.complex128)
        if PBCs:
            M[0, -1] = -2
        zero_block = np.zeros_like(M)
        H = 0.5j * np.block([[zero_block, M], [-M.T, zero_block]])
        return H

    def kitaev_GS_covariance(self, N=None, mu=3.0, DW=False, PBCs=True):
        """
        Diagonalize Kitaev Hamiltonian and return ground-state Majorana covariance.
        """
        H = self.kitaev_majorana_hamiltonian(N=N, mu=mu, DW=DW, PBCs=PBCs)
        eigvals, eigvecs = np.linalg.eigh(H)
        sign_D = np.diag(np.sign(eigvals))
        G = 1j * eigvecs @ sign_D @ eigvecs.conj().T
        return G

    # --------------------------- Measurement Updates (block method) ---------------------------

    def measure_top_layer(self, G, Pi_1, Pi_2, pos_parity=True, Asymm=True):
        """
        Parity measurement on the top (physical) layer using block inversion.
        Pi_1, Pi_2 are (2N,) vectors in the top layer subspace.
        """
        N = self.Nx
        G = np.asarray(G, dtype=np.complex128)
        Pi_1 = np.asarray(Pi_1, dtype=np.complex128).reshape(-1, 1)
        Pi_2 = np.asarray(Pi_2, dtype=np.complex128).reshape(-1, 1)

        Id = np.eye(2*N, dtype=np.complex128)
        H = Pi_1 @ Pi_2.T - Pi_2 @ Pi_1.T
        P = Pi_1 @ Pi_1.T + Pi_2 @ Pi_2.T

        Gtt = self.get_top_layer(G)
        Gbb = self.get_bottom_layer(G)
        Gbt = G[2*N:, :2*N]

        if pos_parity:
            Psi_11, Psi_12, Psi_22 = -H, (Id - P), H
        else:
            Psi_11, Psi_12, Psi_22 = H, (Id - P), -H

        A = self.block_diag(Psi_22, Gbb)
        K = self.block_diag(Psi_12, Gbt)
        M = np.block([[Psi_11, Id], [-Id, Gtt]])
        B = self.solve_reg(M, K.T)

        Gprime = A + K @ B
        if Asymm:
            Gprime = 0.5 * (Gprime - Gprime.T)
        return Gprime

    def measure_bottom_layer(self, G, Pi_1, Pi_2, pos_parity=True, Asymm=True):
        """
        Parity measurement on the bottom (ancilla) layer using block inversion.
        Pi_1, Pi_2 are (2N,) vectors in the bottom layer subspace.
        """
        N = self.Nx
        G = np.asarray(G, dtype=np.complex128)
        Pi_1 = np.asarray(Pi_1, dtype=np.complex128).reshape(-1, 1)
        Pi_2 = np.asarray(Pi_2, dtype=np.complex128).reshape(-1, 1)

        Id = np.eye(2 * N, dtype=np.complex128)
        H = Pi_1 @ Pi_2.T - Pi_2 @ Pi_1.T
        P = Pi_1 @ Pi_1.T + Pi_2 @ Pi_2.T

        Gtt = self.get_top_layer(G)
        Gbb = self.get_bottom_layer(G)
        Gtb = G[:2*N, 2*N:]

        if pos_parity:
            Psi_11, Psi_21, Psi_22 = -H, -(Id - P), H
        else:
            Psi_11, Psi_21, Psi_22 = H, -(Id - P), -H

        A = self.block_diag(Gtt, Psi_22)
        K = self.block_diag(Gtb, Psi_21)
        M = np.block([[Gbb, Id], [-Id, Psi_11]])
        B = self.solve_reg(M, K.T)

        Gprime = A + K @ B
        if Asymm:
            Gprime = 0.5 * (Gprime - Gprime.T)
        return Gprime

    # --------------------------- fSWAP & Operations ---------------------------

    def fSWAP(self, Pi_1_top, Pi_2_top, Pi_1_bot, Pi_2_bot):
        """
        Construct the fSWAP orthogonal matrix swapping a top pair with a bottom pair.
        Inputs are (2N,) in their respective layers; we pad to (4N,) internally.
        """
        Pi_1_top = np.asarray(Pi_1_top, dtype=np.complex128).reshape(-1)
        Pi_2_top = np.asarray(Pi_2_top, dtype=np.complex128).reshape(-1)
        Pi_1_bot = np.asarray(Pi_1_bot, dtype=np.complex128).reshape(-1)
        Pi_2_bot = np.asarray(Pi_2_bot, dtype=np.complex128).reshape(-1)

        N = self.Nx
        zeros = np.zeros(2 * N, dtype=np.complex128)

        # pad to full (4N,)
        Pi_1t = np.concatenate([Pi_1_top, zeros])
        Pi_2t = np.concatenate([Pi_2_top, zeros])
        Pi_1b = np.concatenate([zeros, Pi_1_bot])
        Pi_2b = np.concatenate([zeros, Pi_2_bot])

        Pt = np.outer(Pi_1t, Pi_1t.conj()) + np.outer(Pi_2t, Pi_2t.conj())
        Pb = np.outer(Pi_1b, Pi_1b.conj()) + np.outer(Pi_2b, Pi_2b.conj())
        P = Pt + Pb

        Htb = np.outer(Pi_1t, Pi_1b.conj()) + np.outer(Pi_2t, Pi_2b.conj())
        X = Htb - Htb.T

        Id = np.eye(4 * N, dtype=np.complex128)
        O = Id - P - X
        return O

    def top_layer_meas_feedback(self, G, R):
        '''Adaptive measurement + feedback at site R.'''
        G = np.asarray(G, dtype=np.complex128)
        N = self.Nx

        Pi_1 = np.array(self.MW1[:, R], dtype=np.complex128, copy=True)  # (2N,)
        Pi_2 = np.array(self.MW2[:, R], dtype=np.complex128, copy=True)  # (2N,)

        # local majorana modes
        e1 = np.zeros(2*N, dtype=np.complex128)
        e2 = np.zeros(2*N, dtype=np.complex128)
        e1[R] = 1.0       # Bottom layer, species 1
        e2[N + R] = 1.0   # Bottom layer, species 2

        Gtt = self.get_top_layer(G)
        expectation = np.real(np.dot(Pi_1, Gtt @ Pi_2))
        p_pos = 0.5 * (1.0 + expectation)
        p_pos = np.clip(p_pos, 0.0, 1.0)

        outcome_pos = True if (p_pos >= 1.0 or np.random.rand() < p_pos) else False
        G = self.measure_top_layer(G, Pi_1, Pi_2, pos_parity=outcome_pos)
        if not outcome_pos:
            O = self.fSWAP(Pi_1, Pi_2, e1, e2)
            G = O.T @ G @ O

        return G

    def measure_all_bottom_modes(self, G):
        '''Measure every bottom-layer mode once using fast updates.'''
        G = np.asarray(G, dtype=np.complex128)
        N = self.Nx
        Gbb = self.get_bottom_layer(G)

        for R in range(N):
            e1 = np.zeros(2*N, dtype=np.complex128)
            e2 = np.zeros(2*N, dtype=np.complex128)
            e1[R] = 1.0
            e2[N + R] = 1.0
            
            # Fast Probability extraction (G[u_idx, v_idx] for sparse vectors)
            expectation = float(np.real(Gbb[R, N + R]))
            p_pos = 0.5 * (1.0 + expectation)
            p_pos = np.clip(p_pos, 0.0, 1.0)

            outcome_pos = True if (p_pos >= 1.0 or np.random.rand() < p_pos) else False
            # pass bottom-layer components (2N-length) to the update
            G = self.measure_bottom_layer(G, e1.copy(), e2.copy(), pos_parity=outcome_pos)

        return G

    def post_selection_top_layer(self, G, R):
        '''Project top layer at R onto positive parity.'''
        N = self.Nx
        Pi_1 = self.MW1[:, R]
        Pi_2 = self.MW2[:, R]
        
        # top-layer projection only
        G = self.measure_top_layer(G, Pi_1, Pi_2, pos_parity=True)
        return G

    def randomize_bottom_layer(self, G):
        '''Apply random unitary to bottom layer.'''
        G = np.asarray(G, dtype=np.complex128)
        N = self.Nx
        Il = np.eye(2*N, dtype=np.complex128)
        O_bott = self.random_orthogonal(N)
        O_tot = self.block_diag(Il, O_bott)
        return O_tot.T @ G @ O_tot

    # ----------------------- Adaptive Circuit Driver -----------------------

    def run_adaptive_circuit(
        self,
        G_history=True,
        progress=True,
        cycles=20,
        postselect=False,
        samples=None,
        n_jobs=None,
        backend="loky",
        parallelize_samples=False,
        store="none",
        init_mode="random_pure",
        G_init=None,
        remember_init=True,
        save=True,
        save_suffix=None,
        project_per_cycle=False,
    ):
        """Execute the adaptive circuit."""
        N = self.Nx
        cycles = 5 if cycles is None else int(cycles)
        if store != "none" and not G_history:
            raise ValueError("To store histories, set G_history=True.")

        if not hasattr(self, "MW1") or not hasattr(self, "MW2"):
            self.construct_MW_projectors(nshell=self.nshell)

        def _cache_key(*, N, cycles, samples, nshell, DW, init_mode, store_mode, mu_1, mu_2):
            nsh = "None" if nshell is None else str(nshell)
            if self.DW:
                return (f"N{int(N)}_C{int(cycles)}_S{int(samples)}_nsh={nsh}"
                    f"_DW{int(bool(DW))}_mu1={mu_1}_mu2={mu_2}"
                    f"_init-{init_mode}_store-{store_mode}")
            else:
                return (f"N{int(N)}_C{int(cycles)}_S{int(samples)}_nsh={nsh}"
                    f"_DW{int(bool(DW))}_mu1={mu_1}"
                    f"_init-{init_mode}_store-{store_mode}")

        def _save_histories(array, samples_count):
            if not (save and store != "none" and array is not None): return None
            outdir = self._ensure_outdir("cache/G_history_samples")
            key = _cache_key(N=self.Nx, cycles=cycles, samples=samples_count,
                            nshell=self.nshell, DW=self.DW, init_mode=init_mode, store_mode=store,
                            mu_1=self.mu_1, mu_2=self.mu_2)
            filename = f"{key}.npz"
            if save_suffix:
                root, ext = os.path.splitext(filename)
                filename = f"{root}{save_suffix}{ext}"
            path = os.path.join(outdir, filename)
            np.savez_compressed(path, G_hist=np.asarray(array, dtype=np.complex128))
            return path

        def _make_G0():
            if G_init is not None:
                return np.array(G_init, dtype=np.complex128, copy=True)
            if init_mode == "random_pure":
                return self.init_random_pure_top_layer()
            if init_mode == "maxmix":
                return self.init_max_mix_top_layer()
            raise ValueError("init_mode must be 'random_pure' or 'maxmix'.")

        # ---------------- Single Trajectory ----------------
        if not parallelize_samples or (samples is None or int(samples) <= 1):
            self.G = _make_G0()
            if G_history:
                self.G_list = []
                if remember_init: self.G_list.append(self.G.copy())

            total_sites = cycles * N
            bar = (tqdm_nb.tqdm if not sys.stdout.isatty() else tqdm)
            pbar = bar(total=total_sites, desc="Running adaptive circuit (sites):", unit="site", leave=True) if progress else None

            t_start = time.time()
            for _c in range(cycles):
                for R in range(N):
                    self.G = (self.top_layer_meas_feedback(self.G, R)
                              if not postselect else
                              self.post_selection_top_layer(self.G, R))
                    if pbar is not None: pbar.update(1)
                if not postselect:
                    self.G = self.randomize_bottom_layer(self.G)
                    self.G = self.measure_all_bottom_modes(self.G)
                if project_per_cycle:
                    K = 1j * self.G
                    eigvals, eigvecs = np.linalg.eigh(K)
                    eigvals_clipped = np.sign(eigvals)
                    eigvals_clipped[eigvals_clipped == 0] = 1.0
                    K_proj = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.conj().T
                    self.G = -1j * K_proj
                if G_history:
                    self.G_list.append(self.G.copy())

            if pbar is not None: pbar.close()
            if progress:
                elapsed = time.time() - t_start
                print(f"Total elapsed: {elapsed:.2f} s", flush=True)

            if store == "none": return None
            Nlayer = 2 * N
            per_cycle = self.G_list if (G_history and len(self.G_list) > 0) else [self.G]
            if store == "top":
                hist = [self.get_top_layer(Gk) for Gk in per_cycle]
            elif store == "full":
                hist = [np.asarray(Gk) for Gk in per_cycle]
            else:
                raise ValueError("store must be 'none', 'top', or 'full'")
            
            G_hist = np.expand_dims(np.stack(hist, axis=0), axis=0)
            if store == "full": self.G_history_samples = G_hist
            saved_path = _save_histories(G_hist, samples_count=1)
            return {"G_hist": G_hist, "samples": 1, "save_path": saved_path}

        # ---------------- Parallel Multi-Sample ----------------
        if store == "none":
            raise ValueError("When parallelizing samples, set store='top' or 'full'.")
        samples = 1 if samples is None else int(samples)
        S = samples
        ss = np.random.SeedSequence()
        seeds = ss.generate_state(S, dtype=np.uint32).tolist()
        Nlayer = 2 * N

        def _worker(seed_u32):
            with threadpool_limits(limits=1):
                np.random.seed(int(seed_u32) & 0xFFFFFFFF)
                child = self._spawn_for_parallel()
                child.G0 = _make_G0()
                child.run_adaptive_circuit(
                    G_history=True, progress=False, cycles=cycles, postselect=postselect,
                    parallelize_samples=False, store=store, init_mode=init_mode,
                    remember_init=remember_init, save=False
                )
                full_hist = [np.asarray(Gk) for Gk in child.G_list]
                if store == "full":
                    return np.stack(full_hist, axis=0)
                elif store == "top":
                    top_hist = [self.get_top_layer(Gk) for Gk in full_hist]
                    return np.stack(top_hist, axis=0)
        
        with self._joblib_tqdm_ctx(S, "samples"):
            if backend not in ("loky", "threading"):
                raise ValueError("backend must be 'loky' or 'threading'.")
            if backend == "loky":
                with parallel_backend("loky", n_jobs=n_jobs, inner_max_num_threads=1):
                    with threadpool_limits(limits=1):
                        G_hist_list = Parallel(n_jobs=n_jobs)(
                            delayed(_worker)(seeds[i]) for i in range(S)
                        )
            else:
                os.environ.setdefault("OMP_NUM_THREADS", "1")
                os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
                os.environ.setdefault("MKL_NUM_THREADS", "1")
                os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
                with threadpool_limits(limits=1):
                    G_hist_list = Parallel(n_jobs=n_jobs, backend="threading")(
                        delayed(_worker)(seeds[i]) for i in range(S)
                    )

        G_hist = np.stack(G_hist_list, axis=0)
        self.G_history_samples = G_hist if store == "full" else None
        saved_path = _save_histories(G_hist, samples_count=S)
        return {"G_hist": G_hist, "samples": S, "save_path": saved_path}

    def _spawn_for_parallel(self):
        """Create a lightweight worker instance."""
        child = object.__new__(self.__class__)
        child.Nx = self.Nx
        child.Nlayer_dofs = self.Nlayer_dofs
        child.Ntot_dofs = self.Ntot_dofs
        child.nshell = self.nshell
        child.DW = self.DW
        child.mu_1 = self.mu_1
        child.mu_2 = self.mu_2
        child.time_init = self.time_init
        child.DW_loc = getattr(self, "DW_loc", None)
        child.mu_profile = getattr(self, "mu_profile", None)
        child.MW1 = self.MW1
        child.MW2 = self.MW2
        child.J = self.J
        child.G0 = None
        child.G = None
        child.G_list = []
        child.G_history_samples = None
        return child

# --------------------------- Entanglement & Mutual Info ---------------------------

    def _von_neumann_entropy(self, G_sub):
        """
        Compute Von Neumann entropy S = -Tr(rho ln rho) for a subsystem 
        defined by the Majorana covariance matrix G_sub.
        """
        # 1. Construct Hermitian matrix K = i * G_sub
        # G_sub is real antisymmetric, so K is Hermitian.
        K = 1j * np.asarray(G_sub)
        
        # 2. Diagonalize to find eigenvalues nu
        # Eigenvalues come in pairs +/- nu_k. We only need the positive ones.
        evals = np.linalg.eigvalsh(K)
        
        # 3. Positive modes (take magnitude to guard numerical sign noise)
        n_modes = evals.shape[0] // 2
        nus = np.abs(evals[-n_modes:])
        nus = np.clip(nus, 0.0, 1.0)
        
        # 4. Binary Entropy Formula
        # S = - sum_k [ p_k ln p_k + (1-p_k) ln (1-p_k) ]
        # where p_k = (1 + nu_k) / 2
        p = 0.5 * (1.0 + nus)
        
        # Safe log function (0*log(0) = 0)
        def safe_entr(x):
            return -x * np.log(x + 1e-15)
            
        entropies = safe_entr(p) + safe_entr(1.0 - p)
        return np.sum(entropies)

    def compute_antipodal_MI(self, G):
        """
        Compute Mutual Information I(A:B) between two antipodal regions 
        of size N/4 on the physical (top) layer.
        
        Region A: Sites [0, ..., N/4 - 1]
        Region B: Sites [N/2, ..., N/2 + N/4 - 1]
        """
        N = self.Nx
        L_sub = N // 4
        
        # 1. Isolate Physical (Top) Layer
        # Your basis is [gamma^1_0...gamma^1_{N-1}, gamma^2_0...gamma^2_{N-1}]
        if G.shape[0] == 2 * N:
            G_top = np.asarray(G, dtype=np.complex128)
        else:
            G_top = self.get_top_layer(G) # Shape (2N, 2N)
        
        # 2. Define Lattice Site Indices
        sites_A = np.arange(0, L_sub)
        sites_B = np.arange(N // 2, N // 2 + L_sub)
        
        # 3. Convert Site Indices to Majorana Indices
        # For site i, indices are i (species 1) and N+i (species 2)
        indices_A = np.concatenate([sites_A, sites_A + N])
        indices_B = np.concatenate([sites_B, sites_B + N])
        indices_AB = np.concatenate([indices_A, indices_B])
        
        # 4. Extract Sub-matrices using np.ix_
        G_A = G_top[np.ix_(indices_A, indices_A)]
        G_B = G_top[np.ix_(indices_B, indices_B)]
        G_AB = G_top[np.ix_(indices_AB, indices_AB)]
        
        # 5. Compute Entropies
        S_A = self._von_neumann_entropy(G_A)
        S_B = self._von_neumann_entropy(G_B)
        S_AB = self._von_neumann_entropy(G_AB)
        
        # 6. Mutual Information
        # I(A:B) = S(A) + S(B) - S(AB)
        return S_A + S_B - S_AB
