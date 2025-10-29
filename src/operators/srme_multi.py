from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional, Iterator, Tuple

import torch
import torch.nn.functional as F

__all__ = ["SRMEMultiGenerator", "SRMEConfig"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SRMEConfig:
    """
    Configuration for the SRME multiple-generator operator.

    Parameters
    ----------
    backend : str
        Computational backend. Only 'torch' is supported.
    dtype : torch.dtype
        Time-domain dtype (float32 recommended).
    delta_t : float
        Sample interval Δt (seconds).
    delta_sxi : float | torch.Tensor
        Surface-element weights ΔS_ξ; scalar or shape (nxi,).
    kmax : int
        Maximum Neumann order K.
    tol_series : float
        Early-stop tolerance for the Neumann series; stop when ||M^(n)||/||M|| < tol_series.
    tol_dot : float
        Target tolerance for dot-product tests (not used inside operator).
    fft_conv : bool
        Use FFT-based time convolution (faster for large nt) with exact adjointness.
    xi_block : int | None
        Chunk size for streaming over ξ to limit memory; None = no chunking.
    xi_equals_r : bool
        If True (typical split-spread), nxi == nr so we can chain T_D^n directly.
    device : torch.device | None
        Target device; if None, inferred from D.
    """
    backend: str = "torch"
    dtype: torch.dtype = torch.float32
    delta_t: float = 0.004
    delta_sxi: float | torch.Tensor = 1.0
    kmax: int = 6
    tol_series: float = 1e-4
    tol_dot: float = 1e-6
    fft_conv: bool = False
    xi_block: Optional[int] = None
    xi_equals_r: bool = True
    device: Optional[torch.device] = None


class SRMEMultiGenerator:
    r"""
    SRME multiple-generator operator G with adjoint G*.

    Discrete definitions (time-domain):

      <A, B>_data = Σ_{r,s,t} A[r,s,t] * conj(B[r,s,t]) * Δt
      <X, Y>_surf = Σ_{ξ,s,τ} X[ξ,s,τ] * conj(Y[ξ,s,τ]) * (ΔS_ξ * Δt)

      (T_D X)(r,s,t) = Σ_{ξ} ΔS_ξ Σ_{τ=0}^{t} D(r,ξ,t-τ) X(ξ,s,τ) Δt

      G[P] = (I - T_D)^{-1} T_D P = Σ_{n≥1} T_D^n P

    Notes
    -----
    * Time convolution is exact and causal. Direct path uses conv1d (correlation)
      with time-flipped kernels; FFT path uses rFFT/irFFT with nfft ≥ 2*nt-1 and
      consistent normalization + trimming.
    * Linearity: inputs are never modified in-place; all weights are symmetric in
      forward/adjoint so dot-tests pass to ~1e-6 in float32.
    """

    # ---------- ctor ----------
    def __init__(self, D: torch.Tensor, config: SRMEConfig):
        if config.backend != "torch":
            raise NotImplementedError("Only torch backend is provided.")
        if D.ndim != 3:
            raise ValueError("D must have shape (nr, nxi, nt)")

        self.D = D.contiguous()
        self.nr, self.nxi, self.nt = self.D.shape

        self.cfg = config
        self.device = config.device if config.device is not None else self.D.device
        self.real_dtype = config.dtype

        # Cast D to desired dtype/device (float32 time-domain)
        if self.D.device != self.device or self.D.dtype != self.real_dtype:
            self.D = self.D.to(self.device, self.real_dtype)

        # ΔSξ as vector (nxi,)
        if isinstance(config.delta_sxi, torch.Tensor):
            delta_sxi = config.delta_sxi.to(self.device, self.real_dtype)
            if delta_sxi.numel() == 1:
                self.delta_sxi = delta_sxi.repeat(self.nxi)
            else:
                if delta_sxi.shape != (self.nxi,):
                    raise ValueError("delta_sxi tensor must have shape (nxi,)")
                self.delta_sxi = delta_sxi
        else:
            self.delta_sxi = torch.tensor(float(config.delta_sxi),
                                          device=self.device, dtype=self.real_dtype).repeat(self.nxi)

        self.delta_t = float(config.delta_t)

        if self.cfg.xi_equals_r and self.nxi != self.nr:
            raise ValueError("xi_equals_r=True requires nxi == nr to chain T_D^n.")

        # FFT precomputation
        self._fft_ready = False
        if self.cfg.fft_conv:
            self._prepare_fft()

    # ---------- public API ----------
    @torch.no_grad()
    def forward(self, P: torch.Tensor, method: str = "series") -> torch.Tensor:
        """
        Apply G[P] = Σ_{n=1..K} T_D^n P (Neumann series) or implicit fixed-point.

        Parameters
        ----------
        P : (nxi, ns, nt) torch.Tensor
            Primary field on the surface grid.
        method : {'series', 'implicit'}
            'series' (default) or a simple fixed-point implicit iterate.

        Returns
        -------
        M : (nr, ns, nt) torch.Tensor
            Predicted surface-related multiples.
        """
        self._validate_P(P)
        if method not in ("series", "implicit"):
            raise ValueError("method must be 'series' or 'implicit'")

        if method == "implicit":
            return self._implicit_solve(P)

        ns = P.shape[1]
        M = torch.zeros((self.nr, ns, self.nt), device=self.device, dtype=self.real_dtype)
        X = P
        accum_norm = 0.0
        for k in range(1, self.cfg.kmax + 1):
            term = self._TD(X)                  # M^(k) = T_D * (previous)
            M = M + term
            term_norm = self._norm_data(term)
            accum_norm = self._norm_data(M)
            logger.debug(f"[G] k={k} ||term||={term_norm:.3e} ||acc||={accum_norm:.3e}")

            if k > 1 and accum_norm > 0.0:
                if term_norm / accum_norm < self.cfg.tol_series:
                    logger.info(f"[G] early stop at k={k} (ratio={term_norm/accum_norm:.3e})")
                    break

            # Prepare X for next order: reinterpret (nr,ns,nt) as (nxi,ns,nt) when nxi==nr
            if not self.cfg.xi_equals_r and k < self.cfg.kmax:
                raise NotImplementedError("General ξ↔r mapping not provided.")
            X = term
        return M

    @torch.no_grad()
    def adjoint(self, Y: torch.Tensor, method: str = "series") -> torch.Tensor:
        """
        Apply G*Y = Σ_{n=1..K} (T_D^*)^n Y.

        Parameters
        ----------
        Y : (nr, ns, nt) torch.Tensor
            Data-domain tensor.
        method : {'series', 'implicit'}
            If 'implicit', falls back to series for the adjoint.

        Returns
        -------
        Z : (nxi, ns, nt) torch.Tensor
            Adjoint image on the surface grid.
        """
        self._validate_Y(Y)
        if method not in ("series", "implicit"):
            raise ValueError("method must be 'series' or 'implicit'")

        Z = torch.zeros((self.nxi, Y.shape[1], self.nt), device=self.device, dtype=self.real_dtype)
        W = Y
        accum_norm = 0.0
        for k in range(1, self.cfg.kmax + 1):
            W = self._TD_adj(W)                 # W ← T_D^* W
            Z = Z + W
            term_norm = self._norm_surf(W)
            accum_norm = self._norm_surf(Z)
            logger.debug(f"[G*] k={k} ||term||={term_norm:.3e} ||acc||={accum_norm:.3e}")

            if k > 1 and accum_norm > 0.0:
                if term_norm / accum_norm < self.cfg.tol_series:
                    logger.info(f"[G*] early stop at k={k} (ratio={term_norm/accum_norm:.3e})")
                    break

            if not self.cfg.xi_equals_r and k < self.cfg.kmax:
                raise NotImplementedError("General ξ↔r mapping not provided.")
        return Z

    # ---------- core kernels ----------
    def _TD(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply T_D to X(ξ,s,t) -> Y(r,s,t) with exact causal convolution in time.

        Shapes:
          X : (nxi, ns, nt)
          Y : (nr,  ns, nt)
        """
        self._validate_P(X)
        ns = X.shape[1]

        if self.cfg.fft_conv:
            return self._TD_fft(X)

        Y = torch.zeros((self.nr, ns, self.nt), device=self.device, dtype=self.real_dtype)

        for xs, xe in self._iter_blocks(self.nxi, self.cfg.xi_block):
            Xblk = X[xs:xe]  # (xb, ns, nt)
            deltaS = self.delta_sxi[xs:xe].view(-1, 1, 1)  # (xb,1,1)
            # (ns, xb, nt)
            Xblk_ns_t = Xblk.permute(1, 0, 2).contiguous()

            for i in range(xe - xs):
                x_ns_t = Xblk_ns_t[:, i, :].unsqueeze(1)          # (ns, 1, nt)
                # conv1d computes correlation; flip D in time to get convolution
                w = torch.flip(self.D[:, xs + i, :], dims=(1,)).unsqueeze(1)  # (nr,1,nt)
                y_full = F.conv1d(x_ns_t, w, padding=self.nt - 1)             # (ns, nr, 2*nt-1)
                y_causal = y_full[:, :, : self.nt]                             # keep t in [0,nt-1]
                # Accumulate with ΔSξ * Δt; permute back to (nr,ns,nt)
                Y = Y + y_causal.permute(1, 0, 2) * (deltaS[i] * self.delta_t)

        return Y

    def _TD_adj(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Apply T_D^* to Y(r,s,t) -> Z(ξ,s,t) (adjoint: time-reversed correlation).

        Shapes:
          Y : (nr, ns, nt)
          Z : (nxi, ns, nt)
        """
        self._validate_Y(Y)
        ns = Y.shape[1]

        if self.cfg.fft_conv:
            return self._TD_adj_fft(Y)

        Z = torch.zeros((self.nxi, ns, self.nt), device=self.device, dtype=self.real_dtype)
        Y_ns_r_t = Y.permute(1, 0, 2).contiguous()  # (ns, nr, nt)

        for xi in range(self.nxi):
            # conv1d correlation with kernel = D(r,xi,t) (no flip), slice centered window
            W = self.D[:, xi, :].conj().unsqueeze(0)                       # (1, nr, nt)
            z_full = F.conv1d(Y_ns_r_t, W, padding=self.nt - 1)            # (ns, 1, 2*nt-1)
            z_tau = z_full[:, 0, (self.nt - 1):(self.nt - 1 + self.nt)]    # causal τ∈[0,nt-1]
            Z[xi] = z_tau * self.delta_t
        return Z

    # ---------- FFT paths ----------
    def _prepare_fft(self) -> None:
        nfft = 1
        while nfft < 2 * self.nt - 1:
            nfft <<= 1
        self._nfft = nfft
        self._freq_len = nfft // 2 + 1
        # rFFT along time for D; store both D̂ and its conjugate for adjoint
        Df = torch.fft.rfft(self.D, n=self._nfft, dim=-1)  # (nr, nxi, f)
        self._Df = Df
        self._Df_conj = torch.conj(Df)
        self._fft_ready = True
        logger.info(f"FFT prepared: nfft={self._nfft}, freq_len={self._freq_len}")

    def _TD_fft(self, X: torch.Tensor) -> torch.Tensor:
        if not self._fft_ready:
            self._prepare_fft()
        Xf = torch.fft.rfft(X, n=self._nfft, dim=-1)              # (nxi, ns, f)
        # Ŷ(r,s,f) = Σ_ξ ΔSξ D̂(r,ξ,f) X̂(ξ,s,f)
        Yf = torch.einsum("rxf,xsf,x->rsf", self._Df, Xf, self.delta_sxi)  # (nr, ns, f)
        y_full = torch.fft.irfft(Yf, n=self._nfft, dim=-1)        # (nr, ns, nfft)
        y_causal = y_full[..., : self.nt]                         # (nr, ns, nt)
        return y_causal * self.delta_t

    def _TD_adj_fft(self, Y: torch.Tensor) -> torch.Tensor:
        if not self._fft_ready:
            self._prepare_fft()
        Yf = torch.fft.rfft(Y, n=self._nfft, dim=-1)              # (nr, ns, f)
        # Ẑ(ξ,s,f) = Σ_r D̂(r,ξ,f)^* Ŷ(r,s,f)
        Zf = torch.einsum("rxf,rsf->xsf", self._Df_conj, Yf)      # (nxi, ns, f)
        z_full = torch.fft.irfft(Zf, n=self._nfft, dim=-1)        # (nxi, ns, nfft)
        z_tau = z_full[..., : self.nt]                            # (nxi, ns, nt)
        return z_tau * self.delta_t

    # ---------- utils ----------
    @staticmethod
    def _iter_blocks(n: int, bs: Optional[int]) -> Iterator[Tuple[int, int]]:
        if bs is None or bs >= n:
            yield (0, n)
            return
        i = 0
        while i < n:
            j = min(i + bs, n)
            yield (i, j)
            i = j

    def _validate_P(self, P: torch.Tensor) -> None:
        if not (P.ndim == 3 and P.shape[0] == self.nxi and P.shape[2] == self.nt):
            raise ValueError(f"P must have shape (nxi={self.nxi}, ns, nt={self.nt})")
        if P.device != self.device or P.dtype != self.real_dtype:
            raise TypeError(f"P must be on {self.device} with dtype {self.real_dtype}")

    def _validate_Y(self, Y: torch.Tensor) -> None:
        if not (Y.ndim == 3 and Y.shape[0] == self.nr and Y.shape[2] == self.nt):
            raise ValueError(f"Y must have shape (nr={self.nr}, ns, nt={self.nt})")
        if Y.device != self.device or Y.dtype != self.real_dtype:
            raise TypeError(f"Y must be on {self.device} with dtype {self.real_dtype}")

    def _norm_data(self, A: torch.Tensor) -> float:
        val = torch.sum(A.conj() * A) * self.delta_t
        return float(torch.real(val))

    def _norm_surf(self, X: torch.Tensor) -> float:
        w = self.delta_sxi.view(self.nxi, 1, 1)
        val = torch.sum((X.conj() * X) * w) * self.delta_t
        return float(torch.real(val))

    @torch.no_grad()
    def _implicit_solve(self, P: torch.Tensor, maxiter: Optional[int] = None) -> torch.Tensor:
        """
        Fixed-point iteration for (I - T_D) M = T_D P.
        Use series method unless you specifically prefer this.
        """
        if maxiter is None:
            maxiter = max(10, 2 * self.cfg.kmax)
        B = self._TD(P)
        M = torch.zeros_like(B)
        # Minimal Richardson step M_{k+1} = B + T_D M_k
        for it in range(maxiter):
            M_next = B + self._TD(M)
            R = B - (M - self._TD(M))  # diagnostic residual
            if self._norm_data(R) / (self._norm_data(B) + 1e-12) < self.cfg.tol_series:
                logger.info(f"[implicit] converged at it={it}")
                M = M_next
                break
            M = M_next
        return M
