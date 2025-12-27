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
from pfapack import pfaffian as pf
from IPython.display import clear_output
from matplotlib import pyplot as plt


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

class classD_1dMFGTN:

    def __init__(
        self,
        N,
        DW=True,
        nshell=None,
        envelope_width=None,
        G0=None,
        mu_1=3,
        mu_2=1,
        t_1=1.0,
        t_2=1.0,
    ):
        '''Initialize lattice dimensions, domain-wall option, and starting covariance.'''
        self.time_init = time.time()
        self.Nx = int(N)
        self.Nlayer_dofs = 2 * int(N)
        self.Ntot_dofs = 4 * int(N)
        self.nshell = nshell
        self.envelope_width = envelope_width
        self.DW = bool(DW)
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.t_1 = t_1
        self.t_2 = t_2
        
        # Initialize Domain Wall profile
        if self.DW:
            self.create_domain_wall(mu_1=self.mu_1, mu_2=self.mu_2, t_1=self.t_1, t_2=self.t_2)

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
    
    def random_pure_majorana_covariance(self, N, product_state=False):
        """Build a random 2N x 2N Majorana-basis covariance matrix."""
        rng = np.random.default_rng()
        diag_signs = np.diag(rng.choice([-1, 1], size=N))
        base = np.kron(self.J, diag_signs)        
        if not product_state:
            O = self.random_orthogonal(N)
            cov = O.T @ base @ O
            return cov
        return base
    
    def init_random_pure_top_layer(self):
        """Initialize the top layer as random pure state and bottom layer as random product state."""
        N = self.Nx
        physical_block = self.random_pure_majorana_covariance(N, product_state=False)
        ancilla_block = self.random_pure_majorana_covariance(N, product_state=True)
        cov = self.block_diag(physical_block, ancilla_block)
        return cov 
    
    def init_max_mix_top_layer(self):
        """Initialize the top layer as maximally mixed and bottom layer as random product state."""
        N = self.Nx
        ancilla_block = self.random_pure_majorana_covariance(N, product_state=True)
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

    def project_to_pure_covariance(self, G):
        """
        Project a (possibly noisy) covariance matrix to the nearest pure Gaussian state.
        Returns a skew-symmetric matrix with eigenvalues ±i (i.e., G^2 = -I).
        """
        G = np.asarray(G, dtype=np.complex128)
        G = 0.5 * (G - G.T)  # enforce skew-symmetry
        K = 1j * G
        eigvals, eigvecs = np.linalg.eigh(K)
        eigvals_clipped = np.sign(eigvals)
        eigvals_clipped[eigvals_clipped == 0] = 1.0
        K_proj = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.conj().T
        G_proj = -1j * K_proj
        return G_proj

    def ee_zero_modes(self, G, N_A=None, tol=1e-8):
        """
        Count zero modes in the entanglement spectrum of subsystem A via eigs of i G_A.
        Returns the number of zero mode pairs using a tolerance.
        """
        G = np.asarray(G, dtype=np.complex128)
        G = self.basis_sep_to_interleaved(G)
        if N_A is None:
            N_A = G.shape[0] // 2
        G_A = G[: 2 * N_A, : 2 * N_A]
        eigs = np.linalg.eigvalsh(1j * G_A)
        zero_modes = int(np.sum(np.abs(eigs) < tol) // 2)
        return zero_modes

    # --------------------------- Basis Permutations ---------------------------

    def basis_sep_to_interleaved(self, A):
        """
        Reorders from [g1_all, g2_all] -> [g1_0, g2_0, g1_1, g2_1, ...]
        """
        A = np.asarray(A)
        rows = A.shape[0]
        N_sites = rows // 2
        
        idx = np.empty(rows, dtype=int)
        idx[0::2] = np.arange(N_sites)             # Evens = first half
        idx[1::2] = np.arange(N_sites) + N_sites   # Odds = second half
        
        return A[np.ix_(idx, idx)]

    def basis_interleaved_to_sep(self, A):
        """
        Reorders from [g1_0, g2_0, g1_1, g2_1, ...] -> [g1_all, g2_all]
        """
        A = np.asarray(A)
        rows = A.shape[0]
        N_sites = rows // 2
        
        idx = np.empty(rows, dtype=int)
        idx[:N_sites] = np.arange(0, rows, 2)  # First half = evens
        idx[N_sites:] = np.arange(1, rows, 2)  # Second half = odds
        
        return A[np.ix_(idx, idx)]

    # --------------------------- Coherent Error Model ---------------------------
    def apply_intersite_coherent_errors(self, G, sigma, pbc=True):
        """
        Apply random coherent phase errors (rotations) to the inter-site links.
        
        This rotates the pair (gamma_{i,2}, gamma_{i+1,1}) by angle theta = pi * x,
        where x ~ Uniform[0, sigma]. This disorders the hopping/pairing bonds.
        
        Args:
            G: Covariance matrix.
            sigma: Strength of the error.
            pbc: If True, includes the boundary link (gamma_{N-1,2}, gamma_{0,1}).
        """
        if sigma <= 1e-12: return G
        N = self.Nx
        
        # 1. Identify the indices for the inter-site links
        # In interleaved basis:
        # Site i has gammas at [2*i, 2*i+1]
        # We want to couple: Right gamma of i (2*i+1) <-> Left gamma of i+1 (2*i+2)
        
        if pbc:
            # Connect all N sites in a ring
            # Source: [1, 3, ..., 2N-1] (All gamma_2s)
            idx_src = np.arange(1, 2 * N, 2)
            # Target: [2, 4, ..., 0] (All gamma_1s, rolled to link i to i+1)
            idx_tgt = np.roll(np.arange(0, 2 * N, 2), -1)
        else:
            # Open Chain: N-1 links
            if N < 2: return G
            idx_src = np.arange(1, 2 * N - 1, 2) # [1, 3, ..., 2N-3]
            idx_tgt = np.arange(2, 2 * N, 2)     # [2, 4, ..., 2N-2]
            
        num_links = len(idx_src)

        # 2. Generate Random Rotations
        thetas = np.pi * np.random.uniform(0, sigma, size=num_links)
        c = np.cos(thetas)
        s = np.sin(thetas)

        # 3. Switch to Interleaved Basis
        # Get the active block (top layer) and reorder to [g1_0, g2_0, g1_1, g2_1...]
        G_sub = self.get_top_layer(G)
        G_int = self.basis_sep_to_interleaved(G_sub)

        # 4. Vectorized Update (R^T G R)
        # We apply the rotation mixing rows/cols 'idx_src' and 'idx_tgt'.
        # Using the same rotation definition as your previous function:
        # R = [[c, s], [-s, c]] acting on the subspace (src, tgt).

        # --- Update Rows (Left Multiply by R^T) ---
        # Cache current rows to ensure simultaneous update
        rows_src = G_int[idx_src, :].copy()
        rows_tgt = G_int[idx_tgt, :].copy()
        
        # R^T = [[c, -s], [s, c]]
        # Row_Src' = c * Row_Src - s * Row_Tgt
        # Row_Tgt' = s * Row_Src + c * Row_Tgt
        G_int[idx_src, :] = c[:, None] * rows_src - s[:, None] * rows_tgt
        G_int[idx_tgt, :] = s[:, None] * rows_src + c[:, None] * rows_tgt

        # --- Update Columns (Right Multiply by R) ---
        # Cache current cols (from the row-updated matrix)
        cols_src = G_int[:, idx_src].copy()
        cols_tgt = G_int[:, idx_tgt].copy()
        
        # R = [[c, s], [-s, c]]
        # Col_Src' = c * Col_Src - s * Col_Tgt
        # Col_Tgt' = s * Col_Src + c * Col_Tgt
        G_int[:, idx_src] = c[None, :] * cols_src - s[None, :] * cols_tgt
        G_int[:, idx_tgt] = s[None, :] * cols_src + c[None, :] * cols_tgt

        # 5. Convert back to Separated Basis and Reinsert
        G_top_new = self.basis_interleaved_to_sep(G_int)
        
        G_new = G.copy()
        G_new[:2*N, :2*N] = G_top_new
        
        return G_new
    
    def apply_coherent_errors_slow(self, G, sigma):
        """
        Apply random coherent phase errors to the top layer.
        For each site, rotates the local Majorana pair by angle theta = pi * x,
        where x ~ Uniform[0, sigma].
        
        O_loc = cos(theta) I + sin(theta) * i * sigma_y
              = [[cos, sin], [-sin, cos]]
        """
        if sigma <= 1e-12:
            return G

        N = self.Nx
        G = np.asarray(G)
        
        # Generate random angles for N sites: x ~ Uniform[0, sigma]
        thetas = np.pi * np.random.uniform(0, sigma, size=N)
        c = np.cos(thetas)
        s = np.sin(thetas)
        
        # Construct the local rotation in interleaved basis
        O_err = np.zeros((2 * N, 2 * N), dtype=float)
        O_err[0::2, 0::2] = np.diag(c)
        O_err[1::2, 1::2] = np.diag(c)
        O_err[0::2, 1::2] = np.diag(s)
        O_err[1::2, 0::2] = np.diag(-s)

        # Convert to separated basis so we can apply directly to G
        O_top = self.basis_interleaved_to_sep(O_err)

        if G.shape[0] == 2 * N:
            return O_top.T @ G @ O_top

        if G.shape[0] < 4 * N:
            raise ValueError(f"Covariance too small for two-layer system: got {G.shape}")

        # Act only on the top layer; bottom layer identity
        O_full = self.block_diag(O_top, np.eye(2 * N, dtype=float))
        return O_full.T @ G @ O_full
    
    def apply_coherent_errors(self, G, sigma):
        if sigma <= 1e-12: return G
        N = self.Nx
        
        # 1. Generate Rotations (c, s)
        thetas = np.pi * np.random.uniform(0, sigma, size=N)
        c = np.cos(thetas)
        s = np.sin(thetas)

        # 2. Switch to Interleaved Basis (so site i is at indices 2*i, 2*i+1)
        # This is an O(N^2) copy, much cheaper than O(N^3) multiply
        G_int = self.basis_sep_to_interleaved(self.get_top_layer(G))
        
        # 3. Apply rotation R^T G R explicitly via broadcasting
        # The rotation R_i at site i is [[c, s], [-s, c]]
        # We need to update every 2x2 block G_ij connecting site i and j.
        
        # This is essentially: G'_ij = R_i^T @ G_ij @ R_j
        # We can do this efficiently using np.einsum or just manual row/col mixing
        
        # Vectorized Update Strategy:
        # Update Rows first: G' = R^T G
        # Let's define the odd/even indices
        # 0::2 are gamma_1 (x components), 1::2 are gamma_2 (y components)
        
        # Cache old rows to avoid overwriting while reading
        rows_x = G_int[0::2, :].copy() # gamma_1 rows
        rows_y = G_int[1::2, :].copy() # gamma_2 rows
        
        # Apply R^T on the left (acting on rows)
        # R^T = [[c, -s], [s, c]]
        # New Row X =  c * Old X - s * Old Y
        # New Row Y =  s * Old X + c * Old Y
        
        # We use broadcasting: c is (N,), rows_x is (N, 2N). c[:, None] broadcasts.
        G_int[0::2, :] = c[:, None] * rows_x - s[:, None] * rows_y
        G_int[1::2, :] = s[:, None] * rows_x + c[:, None] * rows_y
        
        # Now apply R on the right (acting on columns)
        # G'' = G' R
        # R = [[c, s], [-s, c]]
        # New Col X = c * Old Col X - s * Old Col Y
        # New Col Y = s * Old Col X + c * Old Col Y
        
        cols_x = G_int[:, 0::2].copy()
        cols_y = G_int[:, 1::2].copy()
        
        # Note: c is shape (N,), but here we broadcasting across rows, so use c[None, :]
        G_int[:, 0::2] = c[None, :] * cols_x - s[None, :] * cols_y
        G_int[:, 1::2] = s[None, :] * cols_x + c[None, :] * cols_y
        
        # 4. Convert back
        G_top_new = self.basis_interleaved_to_sep(G_int)
        
        # Reinsert
        G_new = G.copy()
        G_new[:2*N, :2*N] = G_top_new
        return G_new

    # ------------------ Majorana Wannier (MW) Projectors ------------------

    def create_domain_wall(self, mu_1, mu_2, t_1, t_2):
        '''Construct and store the default domain-wall mass and hopping/pairing profiles.'''
        N = self.Nx
        mu_profile = np.full(N, float(mu_1))
        t_profile = np.full(N, float(self.t_1 if t_1 is None else t_1))
        half = N // 2
        w = max(1, int(np.floor(0.2 * N)))
        x0 = max(0, half - w)
        x1 = min(N, half + w + 1)
        mu_profile[x0:x1] = float(mu_2)
        t_profile[x0:x1] = float(t_2)
        self.DW_loc = [int(x0), int(x1-1)]
        self.DW_mode_loc = [int(x0-1), int(x1-1)]
        self.mu_profile = mu_profile
        self.t_profile = t_profile

    def construct_MW_projectors(self, nshell=None, envelope_width=None, offdiag_tol=1e-12):
        """
        Build 1D Majorana Wannier spinors efficiently using FFT.
        Uses np.vstack to ensure shape (2N, N) per species.

        envelope_width: if provided, apply periodic Gaussian envelope exp(-d^2/(2 b^2))
        with variance b^2 and chord distance d on the ring before normalization.
        """
        N = self.Nx
        if self.DW:
            mu_vec = self.mu_profile
            t_vec = self.t_profile
        else:
            mu_vec = np.full(N, self.mu_1, float)
            t_vec = np.full(N, self.t_1, float)

        k = 2 * np.pi * np.fft.fftfreq(N, d=1.0)              
        Rgrid = np.arange(N, dtype=float)                     

        dy = -2.0 * np.sin(k)[:, None] * t_vec[None, :]       # pairing term ~ |Δ|=t
        dz = -(2.0 * np.cos(k)[:, None] * t_vec[None, :] + mu_vec[None, :])
        dyR = dy                                             # keep (N, N) for broadcasting
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

        # Full complex components before gauge projection
        comp = np.einsum("krm,mn->krn", chi, M, optimize=True)  # (N, N, 2) in x,R,component
        a = comp[..., 0]  # (1,1)·chi
        b = comp[..., 1]  # (1,-1)·chi

        if nshell is not None:
            idx = np.arange(N)
            dist = ((idx[:, None] - Rgrid[None, :] + N // 2) % N) - N // 2
            mask = (np.abs(dist) <= int(nshell)).astype(float)
            a *= mask
            b *= mask
        if envelope_width is not None:
            bwidth = float(envelope_width)
            if bwidth <= 0:
                raise ValueError("envelope_width must be positive.")
            idx = np.arange(N, dtype=float)
            # periodic (chord) distance on a ring: min(|i-j|, N-|i-j|), implemented via centered modulo
            dist = ((idx[:, None] - Rgrid[None, :] + N / 2) % N) - N / 2
            envelope = np.exp(-(dist ** 2) / (2.0 * bwidth * bwidth))
            a *= envelope
            b *= envelope

        if offdiag_tol is not None:
            
            if np.max(np.abs(np.imag(a))) < float(offdiag_tol):
                a = np.real(a) + 0.0j
            else:
                print('off-diagonal 12 is nonzero!')

            if np.max(np.abs(np.imag(b))) < float(offdiag_tol):
                b = np.real(b) + 0.0j
            else:
                print('off-diagonal 21 is nonzero!')

        # Build full MW1/MW2 with off-diagonal mixing when present
        mw1_g1 = np.real(a)
        mw1_g2 = np.imag(b)
        mw2_g1 = -np.imag(a)
        mw2_g2 = np.real(b)

        self.MW1 = np.vstack((mw1_g1.astype(np.float64, copy=True),
                              mw1_g2.astype(np.float64, copy=True)))  # (2N, R)
        self.MW2 = np.vstack((mw2_g1.astype(np.float64, copy=True),
                              mw2_g2.astype(np.float64, copy=True)))  # (2N, R)

        # Normalize each Wannier mode
        norm1 = np.linalg.norm(self.MW1, axis=0, keepdims=True) + 1e-15
        norm2 = np.linalg.norm(self.MW2, axis=0, keepdims=True) + 1e-15
        self.MW1 /= norm1
        self.MW2 /= norm2
  

    # ------------------------------ Real-space Kitaev H and covariance ------------------------------

    def kitaev_majorana_hamiltonian(self, N=None, DW=False, PBCs=True):
        """
        Build the real-space Majorana Hamiltonian H for the Kitaev chain with t=Δ.
        Ordering: [γ1_0..γ1_{N-1}, γ2_0..γ2_{N-1}].
        """
        N = self.Nx if N is None else int(N)
        if DW:
            if not hasattr(self, "mu_profile"):
                self.create_domain_wall(mu_1=self.mu_1, mu_2=self.mu_2, t_1=self.t_1, t_2=self.t_2)
            mu_diag = np.diag(self.mu_profile.astype(np.complex128))
            t_vec = np.asarray(getattr(self, "t_profile", np.full(N, self.t_1, float)), dtype=np.complex128)
            M = -mu_diag
            M += np.diag(-2.0 * t_vec[:-1], k=-1)
        else:
            mu = self.mu_1
            t = self.t_1
            M = -mu * np.eye(N, dtype=np.complex128) + np.diag(-2.0 * t * np.ones(N - 1, dtype=np.complex128), k=-1)
        if PBCs:
            M[0, -1] = -2.0 * (t_vec[-1] if DW else t)
        zero_block = np.zeros_like(M)
        H = 0.5j * np.block([[zero_block, M], [-M.T, zero_block]])
        return H

    def kitaev_GS_covariance(self, N=None, DW=False, PBCs=True):
        """
        Diagonalize Kitaev Hamiltonian and return ground-state Majorana covariance.
        """
        H = self.kitaev_majorana_hamiltonian(N=N, DW=DW, PBCs=PBCs)
        eigvals, eigvecs = np.linalg.eigh(H)
        sign_D = np.diag(np.sign(eigvals))
        G = 1j * eigvecs @ sign_D @ eigvecs.conj().T
        return G
    
    def kitaev_invariant(self, G, tol=1e-16):
        """
        Computes the Z2 topological invariant nu = sgn(Pf(G_interleaved))).
        Returns:
            1 (Trivial)
            -1 (Topological)
        """
        G = self.basis_sep_to_interleaved(G)
        # Pfaffian routine requires a strictly skew-symmetric matrix
        G = 0.5 * (G - G.T)
        pf_val = np.real(pf.pfaffian(G))
        return pf_val

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

    # --------------------------- DoubleBraid & Operations ---------------------------

    def DoubleBraid(self, Pi_1_top, Pi_2_top, Pi_1_bot, Pi_2_bot):
        """
        Construct the DoubleBraid orthogonal matrix swapping a top pair with a bottom pair.
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
            O = self.DoubleBraid(Pi_1, Pi_2, e1, e2)
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
        '''Apply random orthogonal rotation to bottom layer.'''
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
        sigma=None,
        multisample_cycle_threshold=0,
        pre_threshold_progress=None,
        clear_output_each_cycle=False,
        pure_state_tol=1e-12,
    ):
        """Execute the adaptive circuit."""
        N = self.Nx
        cycles = 5 if cycles is None else int(cycles)
        threshold_cycles = cycles // 2 if multisample_cycle_threshold is None else int(multisample_cycle_threshold)
        threshold_cycles = max(0, min(threshold_cycles, cycles))
        use_threshold = threshold_cycles > 0
        if store not in ("none", "top", "full"):
            raise ValueError("store must be 'none', 'top', or 'full'.")

        if not hasattr(self, "MW1") or not hasattr(self, "MW2"):
            self.construct_MW_projectors(nshell=self.nshell, envelope_width=self.envelope_width)

        def _cache_key(*, N, cycles, samples, nshell, DW, init_mode, store_mode, mu_1, mu_2, t_1, t_2, sigma):
            nsh = "None" if nshell is None else str(nshell)
            base = (f"N{int(N)}_C{int(cycles)}_S{int(samples)}_nsh={nsh}"
                    f"_DW{int(bool(DW))}"
                    f"_init-{init_mode}_store-{store_mode}")
            env_val = getattr(self, "envelope_width", None)
            if env_val is not None:
                base = f"{base}_env={float(env_val):.4f}"
            if sigma is not None:
                base = f"{base}_sig{float(sigma):.4f}"
            if self.DW:
                return f"{base}_mu1={mu_1}_mu2={mu_2}_t1={t_1}_t2={t_2}"
            return f"{base}_mu1={mu_1}_t1={t_1}"

        def _save_histories(payload, samples_count):
            """Save arrays with optional suffix; payload is a dict of arrays."""
            if not (save and payload): return None
            outdir = self._ensure_outdir("cache/G_history_samples")
            key = _cache_key(N=self.Nx, cycles=cycles, samples=samples_count,
                            nshell=self.nshell, DW=self.DW, init_mode=init_mode, store_mode=store,
                            mu_1=self.mu_1, mu_2=self.mu_2, t_1=self.t_1, t_2=self.t_2, sigma=sigma)
            filename = f"{key}.npz"
            if save_suffix:
                root, ext = os.path.splitext(filename)
                filename = f"{root}{save_suffix}{ext}"
            path = os.path.join(outdir, filename)
            # convert arrays to complex128 for consistency
            payload_to_save = {k: np.asarray(v, dtype=np.complex128) for k, v in payload.items() if v is not None}
            np.savez_compressed(path, **payload_to_save)
            return path

        def _format_history_list(G_list_local):
            if store == "top":
                return [self.get_top_layer(Gk) for Gk in G_list_local]
            elif store == "full":
                return [np.asarray(Gk) for Gk in G_list_local]
            else:
                raise ValueError("store must be 'none', 'top', or 'full'")

        def _run_cycles(instance, G_start, cycles_to_run, collect_history, show_progress, desc="Running adaptive circuit (sites):"):
            """Run a number of cycles starting from G_start on the provided instance."""
            G_curr = np.array(G_start, dtype=np.complex128, copy=True)
            hist_local = []
            if collect_history and remember_init:
                hist_local.append(G_curr.copy())

            total_sites_local = cycles_to_run * N
            bar_cls = (tqdm_nb.tqdm if not sys.stdout.isatty() else tqdm)
            pbar_local = bar_cls(total=total_sites_local, desc=desc, unit="site", leave=True) if show_progress else None

            for _c in range(cycles_to_run):
                for R in range(N):
                    G_curr = (instance.top_layer_meas_feedback(G_curr, R)
                              if not postselect else
                              instance.post_selection_top_layer(G_curr, R))
                    if pbar_local is not None: pbar_local.update(1)
                if pbar_local is not None:
                    pbar_local.set_postfix(cycle=f"{_c+1}/{cycles_to_run}")
                if not postselect:
                    start_here = time.time()
                    G_curr = instance.randomize_bottom_layer(G_curr)
                    G_curr = instance.measure_all_bottom_modes(G_curr)
                    #print(f'Bottom Layer Reset after cycle {_c+1}, time elapsed = {(time.time() - start_here):.4f}', flush=True)
                if sigma is not None and sigma > 1e-9:
                    #print(f'Applying coherent errors with sigma={sigma:.4f} after cycle {_c+1}.', flush=True)
                    G_curr = instance.apply_coherent_errors(G_curr, sigma)
                deviation = float(np.max(np.abs(G_curr @ G_curr + np.eye(G_curr.shape[0], dtype=G_curr.dtype))))
                if deviation > pure_state_tol:
                    #print(f"pure state deviation max|G^2+Id| = {deviation:.3e}; flattening spectrum.", flush=True)
                    G_curr = instance.project_to_pure_covariance(G_curr)
                if collect_history:
                    hist_local.append(G_curr.copy())
                if clear_output_each_cycle:
                    try:
                        clear_output(wait=True)
                    except Exception:
                        pass

            if collect_history and not hist_local:
                # Ensure at least one entry (final state) when no cycles or init not remembered
                hist_local.append(G_curr.copy())

            if pbar_local is not None: pbar_local.close()
            return G_curr, hist_local

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
            t_start = time.time()
            desc_main = (f"Adaptive circuit | N={N}, cycles={cycles}, nshell={self.nshell}, "
                         f"DW={int(self.DW)}, mu1={self.mu_1:.2f}, mu2={self.mu_2:.2f}, sigma={(sigma if sigma is not None else 0.0):.2f}")
            self.G, hist_single = _run_cycles(self, self.G, cycles_to_run=cycles, collect_history=G_history, show_progress=progress, desc=desc_main)
            self.G_list = hist_single if G_history else []
            if progress:
                elapsed = time.time() - t_start
                print(f"Total elapsed: {elapsed:.2f} s", flush=True)

            if store == "none": return None
            per_cycle = hist_single if (G_history and len(hist_single) > 0) else [self.G]
            hist_formatted = _format_history_list(per_cycle)
            G_hist = np.expand_dims(np.stack(hist_formatted, axis=0), axis=0)
            if store == "full": self.G_history_samples = G_hist
            steady_state_single = self.get_top_layer(self.G) if store == "top" else np.asarray(self.G)
            steady_state = np.expand_dims(steady_state_single, axis=0)  # shape (1, 2N, 2N) or (1, 4N, 4N) for full
            payload = {"G_hist": G_hist} if G_history else {"steady_state": steady_state}
            saved_path = _save_histories(payload, samples_count=1)
            return {
                "G_hist": G_hist if G_history else None,
                "steady_state": steady_state,
                "samples": 1,
                "save_path": saved_path,
            }

        # ---------------- Parallel Multi-Sample ----------------
        if store == "none" and G_history:
            raise ValueError("When parallelizing samples with histories, set store='top' or 'full'.")
        samples = 1 if samples is None else int(samples)
        S = samples
        ss = np.random.SeedSequence()
        seeds = ss.generate_state(S, dtype=np.uint32).tolist()

        # Pre-threshold evolution (shared)
        self.G = _make_G0()
        hist_prefix = []
        pre_prog = progress if pre_threshold_progress is None else bool(pre_threshold_progress)
        if use_threshold:
            if threshold_cycles > 0:
                print(f"Running {threshold_cycles} pre-threshold cycles before branching into {S} samples.", flush=True)
            pre_desc = (f"Pre-threshold | N={N}, cycles={threshold_cycles}, nshell={self.nshell}, "
                        f"DW={int(self.DW)}, mu1={self.mu_1:.2f}, mu2={self.mu_2:.2f}, sigma={(sigma if sigma is not None else 0.0):.2f}")
            self.G, hist_prefix = _run_cycles(
                self,
                self.G,
                cycles_to_run=threshold_cycles,
                collect_history=G_history,
                show_progress=pre_prog and threshold_cycles > 0,
                desc=pre_desc,
            )
        else:
            # No threshold stage; optionally record the initial state if keeping history
            if G_history and remember_init:
                hist_prefix.append(np.array(self.G, copy=True))
        self.G_list = hist_prefix if G_history else []
        if use_threshold:
            if threshold_cycles >= cycles:
                print("Threshold cycles cover requested cycles; skipping multi-sample branching.", flush=True)
            else:
                print(f"Crossed multisample threshold at cycle {threshold_cycles}; spawning parallel samples.", flush=True)

        base_state = self.G.copy()
        prefix_formatted = _format_history_list(hist_prefix) if (G_history and hist_prefix) else []
        remaining_cycles = max(0, cycles - threshold_cycles) if use_threshold else cycles

        def _worker(seed_u32):
            with threadpool_limits(limits=1):
                np.random.seed(int(seed_u32) & 0xFFFFFFFF)
                child = self._spawn_for_parallel()
                child.G0 = base_state.copy()
                child.G = base_state.copy()
                # continue evolution from the shared state
                G_final, hist_local = _run_cycles(
                    child, child.G, cycles_to_run=remaining_cycles, collect_history=G_history, show_progress=False
                )
                if G_history:
                    formatted = _format_history_list(hist_local)
                    return formatted
                # Only need steady-state when not keeping history
                return child.get_top_layer(G_final) if store == "top" else np.asarray(G_final)
        
        sample_desc = (f"samples | N={N}, cycles={cycles}, nshell={self.nshell}, DW={int(self.DW)}, "
                       f"mu1={self.mu_1:.2f}, mu2={self.mu_2:.2f}, sigma={(sigma if sigma is not None else 0.0):.2f}")
        with self._joblib_tqdm_ctx(S, sample_desc):
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

        save_payload = {}
        return_payload = {"samples": S}

        steady_state = None
        if G_history:
            sample_histories = np.stack([np.stack(h_list, axis=0) for h_list in G_hist_list], axis=0) if G_hist_list else None
            # combine prefix and per-sample tails for convenience
            if prefix_formatted:
                prefix_array = np.stack(prefix_formatted, axis=0)
                combined = [np.concatenate([prefix_array, sample_histories[i]], axis=0) for i in range(S)]
                G_hist = np.stack(combined, axis=0)
            else:
                prefix_array = None
                G_hist = sample_histories

            self.G_history_samples = G_hist if (store == "full" and G_hist is not None) else None
            steady_state = G_hist[:, -1] if G_hist is not None else None
            save_payload = {
                "prefix_hist": prefix_array,
                "sample_histories": sample_histories,
                "combined_hist": G_hist,
                "threshold_cycles": np.array(threshold_cycles, dtype=int),
            }
            return_payload.update({"G_hist": G_hist, "prefix_hist": prefix_array})
        else:
            steady_state = np.stack(G_hist_list, axis=0)
            save_payload = {"steady_state": steady_state, "threshold_cycles": np.array(threshold_cycles, dtype=int)}
            return_payload["steady_state"] = steady_state

        if steady_state is not None:
            save_payload["steady_state"] = steady_state
            return_payload["steady_state"] = steady_state

        saved_path = _save_histories(save_payload, samples_count=S)
        return_payload["save_path"] = saved_path
        return return_payload

    def _spawn_for_parallel(self):
        """Create a lightweight worker instance."""
        child = object.__new__(self.__class__)
        child.Nx = self.Nx
        child.Nlayer_dofs = self.Nlayer_dofs
        child.Ntot_dofs = self.Ntot_dofs
        child.nshell = self.nshell
        child.envelope_width = self.envelope_width
        child.DW = self.DW
        child.mu_1 = self.mu_1
        child.mu_2 = self.mu_2
        child.t_1 = self.t_1
        child.t_2 = self.t_2
        child.time_init = self.time_init
        child.DW_loc = getattr(self, "DW_loc", None)
        child.mu_profile = getattr(self, "mu_profile", None)
        child.t_profile = getattr(self, "t_profile", None)
        child.MW1 = self.MW1
        child.MW2 = self.MW2
        child.J = self.J
        child.G0 = None
        child.G = None
        child.G_list = []
        child.G_history_samples = None
        return child

# --------------------------- Entanglement & Observables ---------------------------

    def _von_neumann_entropy(self, G_sub):
        """
        Compute Von Neumann entropy S = -Tr(rho ln rho) for a subsystem 
        defined by the Majorana covariance matrix G_sub.
        """
        K = 1j * np.asarray(G_sub)
        
        evals = np.linalg.eigvalsh(K)
        
        n_modes = evals.shape[0] // 2
        nus = np.abs(evals[-n_modes:])
        nus = np.clip(nus, 0.0, 1.0)
        
        p = 0.5 * (1.0 + nus)
        def safe_entr(x):
            return -x * np.log(x + 1e-15)
            
        entropies = safe_entr(p) + safe_entr(1.0 - p)
        return np.sum(entropies)

    def compute_entanglement_contour(self, G, sites_A, majorana_contour=True):
        """
        Compute the entanglement contour for a subsystem defined by sites_A.
        Returns interleaved Majorana contributions if majorana_contour=True.
        """
        N = self.Nx
        sites_A = np.asarray(sites_A)
        L_sub = len(sites_A)

        # 1. Extract subsystem covariance on the physical layer
        indices_species_1 = sites_A
        indices_species_2 = sites_A + N
        indices_all = np.concatenate([indices_species_1, indices_species_2])

        G_top = self.get_top_layer(G)
        G_sub = G_top[np.ix_(indices_all, indices_all)]

        K = 1j * G_sub
        evals, evecs = np.linalg.eigh(K)
        evals = np.clip(evals, -1.0 + 1e-15, 1.0 - 1e-15)
        p = (1.0 + evals) / 2.0
        h_vals = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)

        U_abs_sq = np.abs(evecs) ** 2
        S_diag = U_abs_sq @ h_vals

        s_gamma1 = 0.5 * S_diag[:L_sub]
        s_gamma2 = 0.5 * S_diag[L_sub:]

        if not majorana_contour:
            return s_gamma1 + s_gamma2
        else:
            contour = np.empty(2 * L_sub, dtype=S_diag.dtype)
            contour[0::2] = s_gamma1
            contour[1::2] = s_gamma2
            return contour

    def compute_antipodal_MI(self, G):
        """
        Compute Mutual Information I(A:B) between two antipodal regions.
        """
        N = self.Nx
        L_sub = N // 4
        
        if G.shape[0] == 2 * N:
            G_top = np.asarray(G, dtype=np.complex128)
        else:
            G_top = self.get_top_layer(G) # Shape (2N, 2N)
        
        sites_A = np.arange(0, L_sub)
        sites_B = np.arange(N // 2, N // 2 + L_sub)
        
        indices_A = np.concatenate([sites_A, sites_A + N])
        indices_B = np.concatenate([sites_B, sites_B + N])
        indices_AB = np.concatenate([indices_A, indices_B])
        
        G_A = G_top[np.ix_(indices_A, indices_A)]
        G_B = G_top[np.ix_(indices_B, indices_B)]
        G_AB = G_top[np.ix_(indices_AB, indices_AB)]
        
        S_A = self._von_neumann_entropy(G_A)
        S_B = self._von_neumann_entropy(G_B)
        S_AB = self._von_neumann_entropy(G_AB)
        
        return S_A + S_B - S_AB

    def compute_defect_density(self, G):
        """
        Compute the domain-wall defect density: D = 1/N sum_X (1 - <i Pi_1(X) Pi_2(X)>)
        """
        N = self.Nx
        
        G_top = self.get_top_layer(G)
        
        G_v = G_top @ self.MW2
        
        total_overlap = np.sum(self.MW1 * G_v)
        
        rho_def = (1.0 - (total_overlap / N))/2
        
        return float(np.real(rho_def))

    def square_correlation_function(self, G, max_X=None):
        """
        Compute C(X) = (1/N) sum_{mu,mu'=0,1} sum_{x=0}^{N-1} |<gamma_{mu,x} gamma_{mu',x+X}>|^2
        for X = 0..(N-1)//2 in the separated basis [g1_0..g1_{N-1}, g2_0..g2_{N-1}].
        """
        N = self.Nx
        if max_X is None:
            max_X = (N - 1) // 2

        if G.shape[0] == 2 * N:
            G_top = np.asarray(G, dtype=np.complex128)
        elif G.shape[0] >= 4 * N:
            G_top = self.get_top_layer(G)
        else:
            raise ValueError(f"Covariance has unexpected shape {G.shape}; expected 2N or >=4N.")

        G_work = np.asarray(G_top, dtype=np.complex128)  # already in separated basis

        x = np.arange(N, dtype=int)
        mu = np.array([0, 1], dtype=int)
        mu_prime = np.array([0, 1], dtype=int)
        X_vals = np.arange(max_X + 1, dtype=int)

        idx_left_base = (mu[:, None] * N + x[None, :]).astype(int)          # (2, N)
        x_shifted = (x[None, :] + X_vals[:, None]) % N                      # (max_X+1, N)
        idx_right_base = (mu_prime[:, None] * N + x_shifted[:, None, :]).astype(int)  # (max_X+1, 2, N)

        idx_left = idx_left_base[None, :, None, :]                          # (1, 2, 1, N)
        idx_right = idx_right_base[:, None, :, :]                           # (max_X+1, 1, 2, N)

        G_vals = G_work[idx_left, idx_right]                                # (max_X+1, 2, 2, N)

        delta = (idx_left == idx_right)
        expect = delta.astype(np.complex128) - 1j * G_vals

        corr = np.sum(np.abs(expect) ** 2, axis=(1, 2, 3)) / float(N)
        return corr


    # --------------------------------------------------------------------------------------
    # Snippet integration (as provided)
    # --------------------------------------------------------------------------------------

    # We're working in the perfect one-site localized limit
    # --- General parity probability for any Majorana pair ---

    def pos_parity_born_prob(self, G_top, Pi_1, Pi_2, renorm=True):
        """
        Born probability p(+) = 0.5 * (1 + Re[Pi_1^T G Pi_2]) for arbitrary Majorana mode pair.
        G_top: (2N, 2N) covariance in separated basis.
        Pi_1, Pi_2: shape (2N,) vectors (need not be local; can be delocalized unpaired modes).
        If renorm=True, the vectors are normalized before use.
        """
        G_top = np.asarray(G_top, dtype=np.complex128)
        Pi_1 = np.asarray(Pi_1, dtype=np.complex128).reshape(-1)
        Pi_2 = np.asarray(Pi_2, dtype=np.complex128).reshape(-1)

        if renorm:
            n1 = np.linalg.norm(Pi_1) + 1e-15
            n2 = np.linalg.norm(Pi_2) + 1e-15
            Pi_1 = Pi_1 / n1
            Pi_2 = Pi_2 / n2

        expectation = float(np.real(Pi_1 @ (G_top @ Pi_2)))
        return float(np.clip(0.5 * (1.0 + expectation), 0.0, 1.0))

    # --- Local measurement helper (still local, but uses the general probability) ---

    def measure_unit_cell_R(self, G_top, R, pos_parity=None, Asymm=True):
        """
        Local parity measurement on (gamma1_R, gamma2_R) at unit cell R for top layer only.
        Returns (Gprime, outcome_bool, p_pos).
        """
        N = self.Nx
        G_top = np.asarray(G_top, dtype=np.complex128)
        if G_top.shape != (2 * N, 2 * N):
            raise ValueError(f"G_top must be (2N,2N); got {G_top.shape}")
        if R < 0 or R >= N:
            raise ValueError(f"R={R} out of range for N={N}")

        e1 = np.zeros(2 * N, dtype=np.complex128); e1[R] = 1.0
        e2 = np.zeros(2 * N, dtype=np.complex128); e2[N + R] = 1.0

        # Born probability (can be non-local if e1/e2 replaced by any Pi_1/Pi_2)
        p_pos = self.pos_parity_born_prob(G_top, e1, e2, renorm=False)
        outcome = pos_parity
        if outcome is None:
            outcome = True if np.random.rand() < p_pos else False

        Pi_1 = e1.reshape(-1, 1); Pi_2 = e2.reshape(-1, 1)
        Id = np.eye(2 * N, dtype=np.complex128)
        H = Pi_1 @ Pi_2.T - Pi_2 @ Pi_1.T
        P = Pi_1 @ Pi_1.T + Pi_2 @ Pi_2.T

        if outcome:
            Psi_11, Psi_12, Psi_22 = -H, (Id - P), H
        else:
            Psi_11, Psi_12, Psi_22 = H, (Id - P), -H

        G_inv = self.solve_reg(G_top, Id)
        middle = self.solve_reg(Psi_22 + G_inv, Id)
        Gprime = Psi_11 + Psi_12 @ middle @ Psi_12.T
        if Asymm:
            Gprime = 0.5 * (Gprime - Gprime.T)
        return Gprime, outcome, p_pos

    # --- DW sweep with tracking (unchanged logic, but probabilities now generalizable) ---

    def sweep_and_append_history(self, G):
        # G should be a DW prepared GS with ideal parameters (mu_1, mu_2, t_1, t_2) = (1, 0, 0, 1)
        N = self.Nx
        start_R = None
        end_R = N // 2
        if G.ndim == 2:
            history = [np.array(G, copy=True)]
        else:
            history = [np.array(G[i], copy=True)]        
        G_curr = history[-1]
        outcomes = []

        tol = 1e-12
        absG = np.abs(G_curr)
        mask = np.triu(np.isclose(absG, 1.0, atol=tol), k=1)
        for i in range(N):
            j = i + N
            if j < 2 * N:
                mask[i, j] = False
        i_idx, j_idx = np.where(mask)
        pairs = [(int(i), int(j)) for i, j in zip(i_idx, j_idx)]
        if not pairs:
            raise ValueError("No DW parity pair found in G.")
        pair_max = max(pairs, key=lambda ij: ij[1] - ij[0])
        i_idx, j_idx = pair_max
        if i_idx >= N:
            raise ValueError(f"Left DW index {i_idx} out of range for N={N}.")

        start_R = max(0, i_idx)
        curr_dw_parity = [float(np.real(G_curr[i_idx, j_idx]))]
        for R in range(start_R, end_R):
            G_curr, outcome, _ = self.measure_unit_cell_R(G_curr, R, Asymm=True)
            outcomes.append(outcome)
            i_idx += 1
            if i_idx >= N:
                break
            curr_dw_parity.append(float(np.real(G_curr[i_idx, j_idx])))
            history.append(np.array(G_curr, copy=True))
        outcome_vals = [np.nan] + [1 if v else -1 for v in outcomes]
        t_steps = np.arange(len(curr_dw_parity))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(t_steps, curr_dw_parity, marker="o", label="MZM Parity")
        ax.plot(t_steps, np.array(outcome_vals), marker="x", linestyle="--", label="Meas. Outcomes")
        ax.set_xlabel("Time step")
        ax.set_ylim(-1.2, 1.2)
        ax.grid(alpha=0.3)
        ax.legend()

        return np.stack(history, axis=0), outcome_vals, curr_dw_parity

    # --- Plot (as before) ---

    def plot_entanglement_contour_heatmap_dw_dynamics(self, G_hist, sites_A=None, save=False, outdir="results", fname="contour_top_only.png"):
        import os
        os.makedirs(outdir, exist_ok=True)

        N = self.Nx
        left_sites = np.arange(0, N // 2)
        right_sites = np.arange(N // 2, N)

        T = G_hist.shape[0]
        contour_left = []
        contour_right = []
        for t in range(T):
            contour_left.append(self.compute_entanglement_contour(G_hist[t], left_sites))
            contour_right.append(self.compute_entanglement_contour(G_hist[t], right_sites))
        contour_left = np.stack(contour_left, axis=0)
        contour_right = np.stack(contour_right, axis=0)

        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 15/2), sharey=True)

        im_l = ax_l.imshow(
            contour_left / np.log(2),
            origin="lower", aspect="auto", cmap="Blues",
            extent=(-0.5, contour_left.shape[1] - 0.5, -0.5, contour_left.shape[0] - 0.5),
        )
        im_r = ax_r.imshow(
            contour_right / np.log(2),
            origin="lower", aspect="auto", cmap="Blues",
            extent=(-0.5, contour_right.shape[1] - 0.5, -0.5, contour_right.shape[0] - 0.5),
        )

        cbar = fig.colorbar(im_l, ax=[ax_l, ax_r])
        cbar.set_label(r"$s_i / \log 2$")

        for ax, contour, title in [
            (ax_l, contour_left, "Left subsystem"),
            (ax_r, contour_right, "Right subsystem"),
        ]:
            ax.set_xlabel("Interleaved Majorana index (γ1_i, γ2_i)")
            ax.set_title(f"{title} | N={self.Nx}")
            ticks = np.arange(0, contour.shape[1], 2)
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(x) for x in ticks])
            ax.set_xticks(np.arange(-0.5, contour.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, contour.shape[0], 1), minor=True)
            ax.grid(which="minor", color="black", alpha=0.4, linewidth=0.5)
            ax.tick_params(which="minor", length=0)

        ax_l.set_ylabel("Time step")
        ax_l.set_yticks(np.arange(contour_left.shape[0]))
        ax_l.set_yticklabels([str(y) for y in range(contour_left.shape[0])])

        right_ticks = np.arange(0, contour_right.shape[1], 2)
        right_labels = [str(N + i) for i in right_ticks]
        ax_r.set_xticks(right_ticks)
        ax_r.set_xticklabels(right_labels)

        if save:
            path = os.path.join(outdir, fname)
            fig.savefig(path, dpi=300, bbox_inches="tight")

        # --- Last time-step: chosen subsystem vs complement ---
        cut_sites = np.arange(20, 32)
        comp_sites = np.setdiff1d(np.arange(N), cut_sites)

        contour_cut = self.compute_entanglement_contour(G_hist[-1], cut_sites)
        contour_comp = self.compute_entanglement_contour(G_hist[-1], comp_sites)

        fig_cut, (ax_cut_l, ax_cut_r) = plt.subplots(1, 2, figsize=(16, 2.5*4/3), sharey=True)

        im_comp = ax_cut_l.imshow(
            contour_comp[None, :] / np.log(2),
            origin="lower", aspect="auto", cmap="Blues",
            extent=(-0.5, contour_comp.shape[0] - 0.5, -0.5, 0.5),
        )
        im_cut = ax_cut_r.imshow(
            contour_cut[None, :] / np.log(2),
            origin="lower", aspect="auto", cmap="Blues",
            extent=(-0.5, contour_cut.shape[0] - 0.5, -0.5, 0.5),
        )

        cbar_cut = fig_cut.colorbar(im_cut, ax=[ax_cut_l, ax_cut_r])
        cbar_cut.set_label(r"$s_i / \log 2$")

        ax_cut_l.set_title("Complement of [20, 32) (final time-step)")
        ax_cut_r.set_title("Cut [20, 32) (final time-step)")
        for ax in (ax_cut_l, ax_cut_r):
            ax.set_xlabel("Interleaved Majorana index (γ1_i, γ2_i)")
            ax.set_yticks([])
            ax.set_xticks(np.arange(-0.5, (contour_comp.shape[0] if ax is ax_cut_l else contour_cut.shape[0]), 1), minor=True)
            ax.grid(which="minor", color="black", alpha=0.4, linewidth=0.5)
            ax.tick_params(which="minor", length=0)

        ticks_comp = np.arange(0, contour_comp.shape[0], 2)
        ax_cut_l.set_xticks(ticks_comp)
        ax_cut_l.set_xticklabels([str(2 * comp_sites[i // 2]) for i in ticks_comp])

        ticks_cut = np.arange(0, contour_cut.shape[0], 2)
        ax_cut_r.set_xticks(ticks_cut)
        ax_cut_r.set_xticklabels([str(2 * (cut_sites[0] + i // 2)) for i in ticks_cut])
