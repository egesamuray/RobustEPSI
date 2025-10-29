## SRME Multiple-Generator operator (Python)

This repo includes a PyTorch implementation of the SRME multiple-generator
\(G = (I - T_D)^{-1} T_D\) and its adjoint \(G^*\).

**Shapes**
- \(D(r,\xi,t)\): `(nr, nxi, nt)` (float32)
- \(P(\xi,s,t)\): `(nxi, ns, nt)` → `G[P] = M(r,s,t)` as `(nr, ns, nt)`

**Inner products** (discrete):
- `<A,B>_data = Σ_{r,s,t} A * conj(B) * Δt`
- `<X,Y>_surf = Σ_{ξ,s,τ} X * conj(Y) * (ΔSξ * Δt)`

**Quick start**
```python
import torch
from src.operators.srme_multi import SRMEMultiGenerator, SRMEConfig

nr, nxi, ns, nt = 64, 64, 32, 1024
dt = 0.004
D = torch.randn(nr, nxi, nt, dtype=torch.float32, device="cuda")

cfg = SRMEConfig(delta_t=dt, delta_sxi=1.0, kmax=6, tol_series=1e-4,
                 fft_conv=True, xi_equals_r=True, device=torch.device("cuda"))
op = SRMEMultiGenerator(D, cfg)

P = torch.randn(nxi, ns, nt, dtype=torch.float32, device="cuda")
M = op.forward(P)           # (nr, ns, nt)
Z = op.adjoint(M)           # (nxi, ns, nt)
Dot-test

bash
Copy code
pytest -q tests/test_srme_multi.py
yaml
Copy code

---

# 8) (Optional) `tests/__init__.py`

```python
# Makes tests a package on some setups; not strictly required.
