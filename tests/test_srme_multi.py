import pytest
import torch

from src.operators.srme_multi import SRMEMultiGenerator, SRMEConfig


def inner_data(A, B, delta_t):
    return torch.real((A.to(torch.float64).conj() * B.to(torch.float64)).sum() * delta_t)


def inner_surf(X, Y, delta_sxi, delta_t):
    w = delta_sxi.view(-1, 1, 1).to(torch.float64)
    return torch.real(((X.to(torch.float64).conj() * Y.to(torch.float64)) * w).sum() * delta_t)


@pytest.mark.parametrize("nt", [32, 64])
@pytest.mark.parametrize("nr,nxi,ns", [(4, 4, 3), (3, 3, 1)])
@pytest.mark.parametrize("fft_conv", [False, True])
def test_td_dottest(nt, nr, nxi, ns, fft_conv):
    torch.manual_seed(31415)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    delta_t = 0.004

    D = torch.randn(nr, nxi, nt, device=device, dtype=torch.float32)
    delta_sxi = (torch.rand(nxi, device=device, dtype=torch.float32) + 0.5)  # positive weights

    cfg = SRMEConfig(delta_t=delta_t, delta_sxi=delta_sxi, kmax=3,
                     fft_conv=fft_conv, xi_equals_r=True, device=device)
    op = SRMEMultiGenerator(D, cfg)

    X = torch.randn(nxi, ns, nt, device=device, dtype=torch.float32)
    Y = torch.randn(nr, ns, nt, device=device, dtype=torch.float32)

    TD_X = op._TD(X)
    TDa_Y = op._TD_adj(Y)

    lhs = inner_data(TD_X, Y, delta_t)
    rhs = inner_surf(X, TDa_Y, delta_sxi, delta_t)
    rel = float(abs(lhs - rhs) / max(1e-12, abs(lhs), abs(rhs)))
    assert rel <= cfg.tol_dot, f"T_D dot-test failed: rel={rel:.3e}"


@pytest.mark.parametrize("nt", [32, 64])
@pytest.mark.parametrize("fft_conv", [False, True])
def test_g_dottest(nt, fft_conv):
    torch.manual_seed(2718)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    nr = nxi = 4
    ns = 2
    delta_t = 0.004

    # choose modest norm so Neumann series behaves well
    D = torch.randn(nr, nxi, nt, device=device, dtype=torch.float32) * 0.2
    delta_sxi = torch.ones(nxi, device=device, dtype=torch.float32)

    cfg = SRMEConfig(delta_t=delta_t, delta_sxi=delta_sxi, kmax=5,
                     tol_series=1e-5, fft_conv=fft_conv, xi_equals_r=True,
                     device=device)
    op = SRMEMultiGenerator(D, cfg)

    P = torch.randn(nxi, ns, nt, device=device, dtype=torch.float32) * 0.3
    Y = torch.randn(nr, ns, nt, device=device, dtype=torch.float32)

    GP = op.forward(P)   # G[P]
    GtY = op.adjoint(Y)  # G*Y

    lhs = inner_data(GP, Y, delta_t)
    rhs = inner_surf(P, GtY, delta_sxi, delta_t)
    rel = float(abs(lhs - rhs) / max(1e-12, abs(lhs), abs(rhs)))
    assert rel <= cfg.tol_dot, f"G dot-test failed: rel={rel:.3e}"


def test_k1_equals_first_order():
    torch.manual_seed(1234)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    nr = nxi = 2
    ns = 1
    nt = 32
    delta_t = 0.004

    D = torch.randn(nr, nxi, nt, device=device, dtype=torch.float32)
    cfg = SRMEConfig(delta_t=delta_t, delta_sxi=1.0, kmax=1, device=device)
    op = SRMEMultiGenerator(D, cfg)

    P = torch.randn(nxi, ns, nt, device=device, dtype=torch.float32)
    M1 = op.forward(P)
    TD_P = op._TD(P)
    assert torch.allclose(M1, TD_P, atol=1e-6), "K=1 must equal first-order multiple"


def test_zero_maps_to_zero():
    device = torch.device("cpu")
    nr = nxi = 2
    ns = 3
    nt = 16

    D = torch.zeros(nr, nxi, nt, device=device, dtype=torch.float32)
    cfg = SRMEConfig(delta_t=0.004, delta_sxi=1.0, kmax=3, device=device)
    op = SRMEMultiGenerator(D, cfg)

    P = torch.zeros(nxi, ns, nt, device=device, dtype=torch.float32)
    Y = torch.zeros(nr, ns, nt, device=device, dtype=torch.float32)

    assert torch.count_nonzero(op.forward(P)) == 0
    assert torch.count_nonzero(op.adjoint(Y)) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_cpu_gpu_parity():
    torch.manual_seed(4321)
    nr = nxi = 3
    ns = 2
    nt = 32
    delta_t = 0.004

    D = torch.randn(nr, nxi, nt, dtype=torch.float32)
    P = torch.randn(nxi, ns, nt, dtype=torch.float32)
    Y = torch.randn(nr, ns, nt, dtype=torch.float32)

    op_cpu = SRMEMultiGenerator(D.clone(), SRMEConfig(delta_t=delta_t, delta_sxi=1.0, kmax=3, device=torch.device("cpu")))
    op_gpu = SRMEMultiGenerator(D.clone().cuda(), SRMEConfig(delta_t=delta_t, delta_sxi=1.0, kmax=3, device=torch.device("cuda")))

    M_cpu = op_cpu.forward(P.clone())
    M_gpu = op_gpu.forward(P.clone().cuda()).cpu()
    assert torch.allclose(M_cpu, M_gpu, atol=5e-6)

    Z_cpu = op_cpu.adjoint(Y.clone())
    Z_gpu = op_gpu.adjoint(Y.clone().cuda()).cpu()
    assert torch.allclose(Z_cpu, Z_gpu, atol=5e-6)
