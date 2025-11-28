import numpy as np
import os
import time
from tqdm import tqdm
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
        mu_triv=3,
        mu_top=1,
    ):
        '''Initialize lattice dimensions, domain-wall option, and starting covariance.'''
        self.time_init = time.time()
        self.Nx = int(N)
        self.Nlayer_dofs = 2 * int(N)
        self.Ntot_dofs = 4 * int(N)
        self.nshell = nshell
        self.DW = bool(DW)
        self.mu_triv = mu_triv
        self.mu_top = mu_top
        
        # Initialize Domain Wall profile
        if self.DW:
            self.create_domain_wall(mu_triv=self.mu_triv, mu_top=self.mu_top)

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
        return tqdm_joblib(tqdm(total=total, desc=desc, unit="task")) if total and total > 1 else nullcontext()
    
    def _ensure_outdir(self, path):
        os.makedirs(path, exist_ok=True)
        return path
    
    def block_diag(self, A, B):
        """Minimal block_diag(A,B) without scipy. Returns [[A,0],[0,B]]."""
        n, m = A.shape[0], B.shape[0]
        Z1 = np.zeros((n, m), dtype=A.dtype)
        Z2 = np.zeros((m, n), dtype=B.dtype)
        return np.block([[A, Z1], [Z2, B]])

    def get_top_layer(self, G):
        """Return the top-layer block of a full covariance."""
        G = np.asarray(G)
        N = G.shape[0] // 2
        return G[:2 * N, :2 * N]

    def get_bottom_layer(self, G):
        """Return the bottom-layer block of a full covariance."""
        G = np.asarray(G)
        N = G.shape[0] // 2
        return G[2 * N:, 2 * N:]

    # ------------------ Majorana Wannier (MW) Projectors ------------------

    def create_domain_wall(self, mu_triv, mu_top):
        '''Construct and store the default domain-wall mass profile.'''
        N = self.Nx
        mu_profile = np.full(N, float(mu_triv))
        half = N // 2
        w = max(1, int(np.floor(0.2 * N)))
        x0 = max(0, half - w)
        x1 = min(N, half + w + 1)
        mu_profile[x0:x1] = float(mu_top)
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
            mu_vec = np.full(N, self.mu_triv, float)

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
        # Top block = Real Part, Bottom block = 0
        self.MW1 = np.vstack((
            Pi[:, :, 0].astype(np.float64, copy=True), 
            np.zeros((N, N))
        )) #(2N, R)
        
        # Top block = 0, Bottom block = Imag Part
        self.MW2 = np.vstack((
            np.zeros((N, N)), 
            Pi[:, :, 1].astype(np.float64, copy=True)
        )) # (2N, R)

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
                self.create_domain_wall(mu_triv=self.mu_triv, mu_top=self.mu_top)
            mu_diag = np.diag(self.mu_profile.astype(np.complex128))
            M = -mu_diag + 2 * np.eye(N, k=1, dtype=np.complex128)
        else:
            M = -mu * np.eye(N, dtype=np.complex128) + 2 * np.eye(N, k=1, dtype=np.complex128)
        if PBCs:
            M[-1, 0] = 2
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

    # --------------------------- Fast O(N^2) Measurement Updates ---------------------------

    def _rank2_update_solver(self, G, u, v, outcome):
        """
        Efficient Rank-2 update for measuring P = i * gamma_u * gamma_v with outcome lambda.
        Formula based on Bravyi (2005) Lagrangian representation, Eq. 33.
        (Sherman-Morrison Formula)
        
        Complexity: O(N^2)
        """
        # Calculate columns of G (Matrix-Vector product: O(N^2) if u dense, O(N) if u sparse)
        Gu = G @ u
        Gv = G @ v
        
        # Calculate overlap (Scalar)
        # Note: For Majorana covariance, G is skew-symmetric. 
        # Expectation value <P> = i * u.T @ G @ v
        # Let omega = u.T @ G @ v
        omega = np.dot(u, Gv)
        
        # Stability check: If already in eigenstate (omega ~ outcome), no update needed
        if abs(outcome - omega) < 1e-9:
            return G
            
        # Denominator 1 - omega^2 (Singular if state is pure and <P> = +/-1)
        denom = 1.0 - omega**2
        
        # If denom is too small, use pseudo-inverse or skip (already projected)
        if abs(denom) < 1e-12:
            return G

        inv_denom = 1.0 / denom
        
        # Coefficients for the update
        # G' = G + c1(Gu Gv^T - Gv Gu^T) + c2(Gu Gu^T + Gv Gv^T)
        # outcome is lambda (+1 or -1)
        c1 = (outcome - omega) * inv_denom
        c2 = (outcome * omega - 1.0) * inv_denom
        
        # Construct update terms (Outer products: O(N^2))
        # Term 1: Antisymmetric part
        T1 = np.outer(Gu, Gv)
        T1 -= T1.T  # Gu Gv^T - Gv Gu^T
        
        # Term 2: Symmetric part
        T2 = np.outer(Gu, Gu) + np.outer(Gv, Gv)
        
        # Update G
        G_new = G + c1 * T1 + c2 * T2
        
        return G_new

    def measure_parity(self, G, u, v, pos_parity=True):
        '''Unified measurement wrapper. Updates G given vectors u, v and desired parity.'''
        outcome = 1.0 if pos_parity else -1.0
        return self._rank2_update_solver(G, u, v, outcome)

    # --------------------------- fSWAP & Operations ---------------------------

    def fSWAP(self, Pi_1_top, Pi_2_top, Pi_1_bot, Pi_2_bot):
        '''Construct the fSWAP unitary. Use concatenate to ensure 1D inputs.
            Furthermore, we will assume that the inputs are size (4N, )
        '''
        # Ensure inputs are 1D
        Pi_1t = np.asarray(Pi_1_top, dtype=np.complex128)
        Pi_2t = np.asarray(Pi_2_top, dtype=np.complex128)
        Pi_1b = np.asarray(Pi_1_bot, dtype=np.complex128)
        Pi_2b = np.asarray(Pi_2_bot, dtype=np.complex128)        


        # Projector P
        Pt = np.outer(Pi_1t, Pi_1t.conj()) + np.outer(Pi_2t, Pi_2t.conj())
        Pb = np.outer(Pi_1b, Pi_1b.conj()) + np.outer(Pi_2b, Pi_2b.conj())
        P = Pt + Pb

        # Swap generator X = H - H.T
        Htb = np.outer(Pi_1t, Pi_1b.conj()) + np.outer(Pi_2t, Pi_2b.conj())
        X = Htb - Htb.T
        
        Id = np.eye(P.shape[0], dtype=np.complex128)
        O = Id - P - X
        return O

    def top_layer_meas_feedback(self, G, R):
        '''Adaptive measurement + feedback at site R.'''
        G = np.asarray(G, dtype=np.complex128)
        N = self.Nx

        # 1. PADDING: Construct Full 4N vectors for Top Layer Wannier modes
        # self.MW1/2 are (2N, N). We extract column R (size 2N).
        # We append 2N zeros for the bottom layer.
        Pi_1 = np.concatenate((self.MW1[:, R], np.zeros(2*N))) # (4N,)
        Pi_2 = np.concatenate((self.MW2[:, R], np.zeros(2*N))) # (4N,)

        # 2. PADDING: Construct Full 4N vectors for Ancilla modes at R
        # Indices: [Top(2N) | Bot(2N)] -> Bot is range [2N, 4N]
        e1 = np.zeros(4*N, dtype=np.complex128)
        e2 = np.zeros(4*N, dtype=np.complex128)
        e1[2*N + R] = 1.0       # Bottom layer, species 1
        e2[2*N + N + R] = 1.0   # Bottom layer, species 2

        # Fast Born Probability: p = (1 + <i g1 g2>) / 2 = (1 + u^T G v) / 2
        expectation = np.dot(Pi_1, G @ Pi_2) # O(N^2)
        p_pos = 0.5 * (1.0 + np.real(expectation))
        p_pos = np.clip(p_pos, 0.0, 1.0)

        # Sample and Update
        if np.random.rand() < p_pos:
            G = self.measure_parity(G, Pi_1, Pi_2, pos_parity=True)
        else:
            G = self.measure_parity(G, Pi_1, Pi_2, pos_parity=False)
            
            # Apply fSWAP correction
            O = self.fSWAP(Pi_1, Pi_2, e1, e2)
            G = O.T @ G @ O

        return G

    def measure_all_bottom_modes(self, G):
        '''Measure every bottom-layer mode once using fast updates.'''
        G = np.asarray(G, dtype=np.complex128)
        N = self.Nx

        for R in range(N):
            # PADDING: Construct vectors in 4N space for bottom layer
            u = np.zeros(4*N, dtype=np.complex128)
            v = np.zeros(4*N, dtype=np.complex128)
            u[2*N + R] = 1.0
            v[2*N + N + R] = 1.0
            
            # Fast Probability extraction (G[u_idx, v_idx] for sparse vectors)
            expectation = G[2*N + R, 3*N + R] 
            p_pos = 0.5 * (1.0 + np.real(expectation))
            p_pos = np.clip(p_pos, 0.0, 1.0)
            
            outcome_pos = (np.random.rand() < p_pos)
            G = self.measure_parity(G, u, v, pos_parity=outcome_pos)

        return G

    def post_selection_top_layer(self, G, R):
        '''Project top layer at R onto positive parity.'''
        N = self.Nx
        # PADDING: Ensure full 4N vectors
        Pi_1 = np.concatenate((self.MW1[:, R], np.zeros(2*N)))
        Pi_2 = np.concatenate((self.MW2[:, R], np.zeros(2*N)))
        
        G = self.measure_parity(G, Pi_1, Pi_2, pos_parity=True)
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
    ):
        """Execute the adaptive circuit."""
        N = self.Nx
        cycles = 5 if cycles is None else int(cycles)
        if store != "none" and not G_history:
            raise ValueError("To store histories, set G_history=True.")

        if not hasattr(self, "MW1") or not hasattr(self, "MW2"):
            self.construct_MW_projectors(nshell=self.nshell)

        def _cache_key(*, N, cycles, samples, nshell, DW, init_mode, store_mode, mu_triv, mu_top):
            nsh = "None" if nshell is None else str(nshell)
            return (f"N{int(N)}_C{int(cycles)}_S{int(samples)}_nsh{nsh}"
                    f"_DW{int(bool(DW))}_mutriv{mu_triv}_mutop{mu_top}"
                    f"_init-{init_mode}_store-{store_mode}")

        def _save_histories(array, samples_count):
            if not (save and store != "none" and array is not None): return None
            outdir = self._ensure_outdir("cache/G_history_samples")
            key = _cache_key(N=self.Nx, cycles=cycles, samples=samples_count,
                            nshell=self.nshell, DW=self.DW, init_mode=init_mode, store_mode=store,
                            mu_triv=self.mu_triv, mu_top=self.mu_top)
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
            pbar = tqdm(total=total_sites, desc="Adaptive (sites)", unit="site", leave=True) if progress else None

            for _c in range(cycles):
                for R in range(N):
                    self.G = (self.top_layer_meas_feedback(self.G, R)
                              if not postselect else
                              self.post_selection_top_layer(self.G, R))
                    if pbar is not None: pbar.update(1)
                if not postselect:
                    self.G = self.randomize_bottom_layer(self.G)
                    self.G = self.measure_all_bottom_modes(self.G)
                if G_history:
                    self.G_list.append(self.G.copy())

            if pbar is not None: pbar.close()

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
                if store == "full": return np.stack(full_hist, axis=0)
                elif store == "top":
                    top_hist = [Gk[:Nlayer, :Nlayer] for Gk in full_hist]
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
        child.mu_triv = self.mu_triv
        child.mu_top = self.mu_top
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
