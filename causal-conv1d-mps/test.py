#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import pytest
import causal_conv1d_mps


def check_gradients_numerical(
    func,
    inputs,
    eps: float = 1e-3,
    atol: float = 1e-3,
    rtol: float = 1e-2,
):
    for inp in inputs:
        if isinstance(inp, torch.Tensor) and inp.requires_grad:
            inp.grad = None
    out = func()
    loss = out.sum() if out.dim() > 0 else out
    loss.backward()
    analytical = [
        t.grad.clone() if isinstance(t, torch.Tensor) and t.grad is not None else None
        for t in inputs
    ]

    numerical = []
    for inp in inputs:
        if not isinstance(inp, torch.Tensor) or not inp.requires_grad:
            numerical.append(None)
            continue
        grad = torch.zeros_like(inp)
        flat = inp.data.view(-1)
        gflat = grad.view(-1)
        base = inp.data.clone().view(-1)
        with torch.no_grad():
            for j in range(flat.numel()):
                flat[j] = base[j] + eps
                out_p = func()
                lp = out_p.sum().item() if out_p.dim() > 0 else out_p.item()
                flat[j] = base[j] - eps
                out_m = func()
                lm = out_m.sum().item() if out_m.dim() > 0 else out_m.item()
                gflat[j] = (lp - lm) / (2 * eps)
                flat[j] = base[j]
        numerical.append(grad)

    all_ok = True
    is_mps = any(
        isinstance(t, torch.Tensor)
        and getattr(t, "device", torch.device("cpu")).type == "mps"
        for t in inputs
    )
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        if a is None and n is None:
            continue
        if a is None or n is None:
            print(f"Gradient mismatch for input {i}: one is None")
            all_ok = False
            continue
        if not is_mps:
            ok = torch.allclose(a, n, atol=atol, rtol=rtol)
        else:
            diff = (a - n).abs()
            max_abs = diff.max().item()
            rel = diff / (n.abs().clamp(min=1e-3))
            median_rel = rel.median().item()
            ok = (max_abs <= 0.08) and (median_rel <= 0.08)
        if not ok:
            print(f"Gradient mismatch for input {i}:")
            print(f"  Analytical: {a.flatten()[:5]}...")
            print(f"  Numerical:  {n.flatten()[:5]}...")
            print(f"  Max diff: {(a - n).abs().max().item()}")
            if is_mps:
                print(
                    f"  Median rel err: {((a - n).abs() / (n.abs().clamp(min=1e-3))).median().item()}"
                )
            all_ok = False
        else:
            print(f"✓ Gradient check passed for input {i}")
    return all_ok


def causal_conv1d_reference(x, weight, bias=None, silu_activation=False):
    batch, dim, seqlen = x.shape
    width = weight.shape[1]

    itype = x.dtype
    x_q = x.detach().cpu().to(dtype=itype)
    weight_q = weight.detach().cpu().to(dtype=itype)
    bias_q = bias.detach().cpu().to(dtype=itype) if bias is not None else None

    x_cpu = x_q.float()
    weight_cpu = weight_q.float()
    bias_cpu = bias_q.float() if bias_q is not None else None

    x_padded = F.pad(x_cpu, (width - 1, 0))

    out = F.conv1d(
        x_padded, weight_cpu.unsqueeze(1), bias=bias_cpu, groups=dim, padding=0
    )

    out = out[:, :, :seqlen]

    if silu_activation:
        out = F.silu(out)

    return out


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("silu_activation", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("width", [4])
@pytest.mark.parametrize("seqlen", [1, 2, 8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("dim", [64, 128, 256])
def test_causal_conv1d_mps(dim, seqlen, width, has_bias, silu_activation, itype):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    device = "mps"
    if itype == torch.float32:
        rtol, atol = (3e-4, 1e-3)
    elif itype == torch.bfloat16:
        rtol, atol = (5e-3, 2e-2)
    else:  # float16
        rtol, atol = (3e-3, 5e-3)

    torch.random.manual_seed(42)
    batch = 2

    x = torch.randn(batch, dim, seqlen, device=device, dtype=itype)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32)
    else:
        bias = None

    out_mps = causal_conv1d_mps.causal_conv1d_fwd(x, weight, bias, silu_activation)

    out_ref = causal_conv1d_reference(x, weight, bias, silu_activation)
    out_ref = out_ref.to(device=device, dtype=itype)

    print(f"Output max diff: {(out_mps - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out_mps - out_ref).abs().mean().item()}")

    assert torch.allclose(out_mps, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32])
@pytest.mark.parametrize("silu_activation", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("width", [4])
@pytest.mark.parametrize("seqlen", [8, 16, 32, 64])
@pytest.mark.parametrize("dim", [64, 128])
def test_short_conv_fused(dim, seqlen, width, has_bias, silu_activation, itype):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    device = "mps"
    if itype == torch.float32:
        rtol, atol = (3e-4, 1e-3)
    elif itype == torch.bfloat16:
        rtol, atol = (5e-3, 2e-2)
    else:
        rtol, atol = (3e-3, 5e-3)

    torch.random.manual_seed(42)
    batch = 2

    x = torch.randn(batch, seqlen, dim, device=device, dtype=itype)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32)
    else:
        bias = None

    attention_mask = torch.ones(batch, seqlen, device=device, dtype=torch.float32)
    for b in range(batch):
        valid_len = torch.randint(seqlen // 2, seqlen, (1,)).item()
        attention_mask[b, valid_len:] = 0

    out_mps = causal_conv1d_mps.short_conv_fused(
        x, weight, bias, attention_mask, activation=silu_activation, residual=True
    )

    x_masked = x * attention_mask.unsqueeze(-1)
    x_transposed = x_masked.transpose(-1, -2).contiguous()

    conv_out = causal_conv1d_reference(x_transposed, weight, bias, silu_activation)
    conv_out = conv_out.transpose(-1, -2).to(device=device, dtype=itype)

    out_ref = x + conv_out  # residual connection

    print(f"Output max diff: {(out_mps - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out_mps - out_ref).abs().mean().item()}")

    assert torch.allclose(out_mps, out_ref, rtol=rtol, atol=atol)


def test_edge_cases():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    device = "mps"

    x = torch.randn(1, 1, 1, device=device, dtype=torch.float32)
    weight = torch.randn(1, 4, device=device, dtype=torch.float32)
    bias = torch.randn(1, device=device, dtype=torch.float32)

    result = causal_conv1d_mps.causal_conv1d_fwd(x, weight, bias, False)
    assert result.shape == (1, 1, 1)

    x = torch.randn(2, 3, 5, device=device, dtype=torch.float32)
    weight = torch.randn(3, 4, device=device, dtype=torch.float32)

    result = causal_conv1d_mps.causal_conv1d_fwd(x, weight, None, False)
    assert result.shape == (2, 3, 5)


def test_error_handling():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    device = "mps"

    x = torch.randn(2, 4, 8, device=device, dtype=torch.float32)
    weight = torch.randn(5, 4, device=device, dtype=torch.float32)

    with pytest.raises(ValueError, match="does not match"):
        causal_conv1d_mps.causal_conv1d_fwd(x, weight, None, False)

    x_2d = torch.randn(4, 8, device=device, dtype=torch.float32)
    weight = torch.randn(4, 4, device=device, dtype=torch.float32)

    with pytest.raises(ValueError, match="Expected 3D input tensor"):
        causal_conv1d_mps.causal_conv1d_fwd(x_2d, weight, None, False)


def test_different_dtypes():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    device = "mps"
    batch, dim, seqlen, width = 2, 64, 32, 4

    x = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32)
    bias = torch.randn(dim, device=device, dtype=torch.float32)

    result = causal_conv1d_mps.causal_conv1d_fwd(x, weight, bias, False)
    assert result.dtype == torch.float32
    assert result.shape == (batch, dim, seqlen)


if __name__ == "__main__":
    pytest.main([__file__])


def test_gradients_causal_conv1d():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    device = torch.device("mps")
    batch_size, dim, seqlen, width = 2, 8, 16, 4
    x = torch.randn(batch_size, dim, seqlen, device=device, requires_grad=True)
    weight = torch.randn(dim, width, device=device, requires_grad=True)
    bias = torch.randn(dim, device=device, requires_grad=True)

    def f():
        return causal_conv1d_mps.causal_conv1d_fn(x, weight, bias, activation="silu")

    ok = check_gradients_numerical(f, [x, weight, bias], eps=1e-3)
    assert ok


def test_gradients_short_conv_fused():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    device = torch.device("mps")
    batch_size, seqlen, dim, width = 2, 8, 16, 4
    x = torch.randn(batch_size, seqlen, dim, device=device, requires_grad=True)
    weight = torch.randn(dim, width, device=device, requires_grad=True)
    bias = torch.randn(dim, device=device, requires_grad=True)
    attention_mask = torch.ones(batch_size, seqlen, device=device)

    def f():
        return causal_conv1d_mps.short_conv_fused_fn(
            x, weight, bias, attention_mask, activation=True, residual=True
        )

    ok = check_gradients_numerical(f, [x, weight, bias], eps=1e-3)
    assert ok


def test_gradients_short_conv_update():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    device = torch.device("mps")
    batch_size, dim, width, state_len = 2, 8, 4, 8
    x = torch.randn(batch_size, dim, device=device, requires_grad=True)
    conv_state = torch.randn(
        batch_size, dim, state_len, device=device, requires_grad=True
    )
    weight = torch.randn(dim, width, device=device, requires_grad=True)
    bias = torch.randn(dim, device=device, requires_grad=True)
    cache_seqlens = torch.randint(
        0, state_len, (batch_size,), device=device, dtype=torch.int32
    )

    def f():
        return causal_conv1d_mps.short_conv_update_fn(
            x,
            conv_state.clone(),
            weight,
            bias,
            cache_seqlens,
            activation=True,
            residual=True,
        )

    ok = check_gradients_numerical(f, [x, weight, bias], eps=1e-3)
    assert ok
