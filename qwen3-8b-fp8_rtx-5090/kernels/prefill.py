"""Triton kernels for prefill (many tokens at once).

Prefill is compute-bound, so it uses tensor cores through `tl.dot`. The math is the
"fine-grained FP8" recipe the checkpoint was published with (W8A8, dynamic activations):

  1. activations are quantized per token, per 128-wide group of the hidden dimension:
         s_x = max(|x_group|) / 448        (448 = largest float8_e4m3fn value)
         q_x = fp8(x / s_x)
  2. weights are already FP8 with one float32 scale per 128x128 block (`weight_scale_inv`)
  3. for every 128-wide K block:  acc += dot(q_x, q_w) * s_x * s_w     (fp32 accumulator)
  4. the result is rounded to bf16

The decode GEMV in `decode.cu` computes exactly the same expression for one token.
"""
import torch
import triton
import triton.language as tl

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


@triton.jit
def _act_quant_kernel(x_ptr, q_ptr, s_ptr, BLOCK: tl.constexpr):
    # One program per 128-element group.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    q = (x / s).to(q_ptr.dtype.element_ty)
    tl.store(q_ptr + offs, q)
    tl.store(s_ptr + pid, s)


def act_quant(x: torch.Tensor, block: int = 128):
    """x: [M, K] bf16 -> (q [M, K] fp8, s [M, K/block] fp32)."""
    assert x.is_contiguous() and x.shape[-1] % block == 0
    q = torch.empty_like(x, dtype=FP8)
    s = torch.empty(*x.shape[:-1], x.shape[-1] // block, dtype=torch.float32, device=x.device)
    _act_quant_kernel[(x.numel() // block,)](x, q, s, BLOCK=block)
    return q, s


@triton.jit
def _fp8_block_matmul_kernel(
    A, B, C, As, Bs, M, N, K,
    stride_am, stride_bn, stride_cm, stride_asm, stride_bsn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # C[M, N] = A[M, K] @ B[N, K]^T with per-block scales; BLOCK_K = 128 = quant block, BLOCK_N <= 128.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] + offs_n[None, :] * stride_bn      # [BLOCK_K, BLOCK_N] view of W^T
    as_ptrs = As + offs_m * stride_asm
    bs_ptrs = Bs + (offs_n // 128) * stride_bsn          # weight scales are per 128 output rows

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        a_s = tl.load(as_ptrs + kb)
        b_s = tl.load(bs_ptrs + kb)
        acc += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=offs_cm[:, None] < M)


@triton.jit
def _fp8_block_matmul_tiled_kernel(
    A, B, C, As, Bs, M, N, K,
    stride_am, stride_cm, stride_asm, stride_bsn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Same as _fp8_block_matmul_kernel, but B is the decode GEMV's K-major mma-tiled layout
    # (model.tile_fp8(w, kmajor=True)), so prefill needs no row-major copy. Element (n, k) with
    # n = 8*ng + g, k = 64*kc + 16*q + 4*t + b sits at kc*64N + ng*512 + g*64 + t*16 + q*4 + b: the
    # address splits into a k part and an n part. Same values, same dot, same K order -> same bits;
    # but only 4-byte runs are contiguous along k, so the loads are narrower (13-30% slower GEMMs).
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :]
    fk = (offs_k // 64) * (64 * N) + ((offs_k % 16) // 4) * 16 + ((offs_k % 64) // 16) * 4 + offs_k % 4
    fk = tl.max_contiguous(tl.multiple_of(fk, 4), 4)
    fn = (offs_n // 8) * 512 + (offs_n % 8) * 64
    b_ptrs = B + fk[:, None] + fn[None, :]
    as_ptrs = As + offs_m * stride_asm
    bs_ptrs = Bs + (offs_n // 128) * stride_bsn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        a_s = tl.load(as_ptrs + kb)
        b_s = tl.load(bs_ptrs + kb)
        acc += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_K
        b_ptrs += (BLOCK_K // 64) * 64 * N

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=offs_cm[:, None] < M)


def fp8_linear(x: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    """x: [M, K] bf16, w: [N, K] fp8 (row-major), w_scale: [N/128, K/128] fp32 -> [M, N] bf16."""
    q, s = act_quant(x.contiguous())
    return fp8_matmul(q, s, w, w_scale)


def fp8_matmul(q: torch.Tensor, s: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor,
               kmajor_tiled: bool = False) -> torch.Tensor:
    """Already-quantized activations q [M, K] fp8 + s [M, K/128] fp32 times w [N, K] -> [M, N] bf16.
    kmajor_tiled: w is in the decode GEMV's K-major tiled layout instead of row-major."""
    M, K = q.shape
    N = w.shape[0]
    assert N % 128 == 0 and K % 128 == 0
    out = torch.empty(M, N, dtype=torch.bfloat16, device=q.device)
    # Tile shape only changes how outputs are split over CTAs, never the order in which one output's
    # K blocks are accumulated, so results are bitwise identical for any choice here. Small M is
    # weight-streaming bound and needs many CTAs. For large M the limit is wave quantization: one
    # 128x128 CTA per SM, and e.g. o_proj at M=1024 has 256 tiles for 170 SMs (1.5 waves -> 2 rounds).
    # Smaller 4-warp tiles fill the SMs better (measured with a tile-size sweep): at M=1024,
    # qkv 128x64, o 64x64, gate_up 128x64, down 64x64 take 11-22% less nsys time than 128x128x8w.
    stages = 3
    if M <= 256:
        block_m, block_n, warps = min(max(triton.next_power_of_2(M), 16), 64), 64, 4
    elif M <= 384:
        block_m, block_n, warps = 64, 64, 4
    elif M < 1024 or N > 4096:
        block_m, block_n, warps = 128, 64, 4
    else:
        block_m, block_n, warps = 64, 64, 4
    grid = (triton.cdiv(M, block_m), N // block_n)
    if kmajor_tiled:
        assert block_n % 8 == 0
        _fp8_block_matmul_tiled_kernel[grid](
            q, w, out, s, w_scale, M, N, K, q.stride(0), out.stride(0), s.stride(0), w_scale.stride(0),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=128, num_warps=warps, num_stages=stages)
        return out
    _fp8_block_matmul_kernel[grid](
        q, w, out, s, w_scale, M, N, K,
        q.stride(0), w.stride(0), out.stride(0), s.stride(0), w_scale.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=128, num_warps=warps, num_stages=stages,
    )
    return out
