// Decode kernels for Qwen3-8B FP8 at batch size 1.
//
// At batch 1 every weight byte is read once per token and used for a single multiply-add, so a
// decode step is limited by DRAM bandwidth, not by math. The kernels below therefore try to do
// three things: (1) keep enough bytes in flight to saturate DRAM, (2) never read anything twice,
// (3) do all the small "glue" work (norms, quantization, residual adds, SwiGLU, argmax) inside the
// big weight-streaming kernels so the GPU does not idle between kernels.
//
// Numerics follow the reference implementation exactly where that is possible:
//   - activations entering an FP8 linear are quantized per 128-wide group: s = max|x|/448, q = fp8(x/s)
//   - per 128-wide K block: acc += dot(q_x, q_w) * s_x * s_w, accumulated in fp32, rounded to bf16
//   - every elementwise op rounds to bf16 after each step, like PyTorch does on bf16 tensors.
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace {

using bf16 = __nv_bfloat16;

constexpr int HIDDEN = 4096;
constexpr int HEAD_DIM = 128;
constexpr int N_Q_HEADS = 32;
constexpr int N_KV_HEADS = 8;
constexpr int GQA = N_Q_HEADS / N_KV_HEADS;  // 4 query heads share one K/V head
constexpr int QBLOCK = 128;                  // quantization block, both weights and activations

// ------------------------------------------------------------------------------------------------
// Small helpers
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// Programmatic dependent launch (PDL). A kernel launched with the PDL attribute may start while
// the previous kernel in the stream is still running. Everything before pdl_wait() must not read
// data produced by earlier kernels (weights are fine: they never change); pdl_wait() blocks until
// the previous kernel has finished and its writes are visible. pdl_trigger() tells the hardware
// the next kernel may now be launched (it will run its own pre-wait part, e.g. weight prefetch).
// Both are no-ops when the kernel is launched without the attribute.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// Optional per-warp timestamps for profiling (null in normal runs).
__device__ unsigned long long* g_timeline = nullptr;
__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}

// Load 16 bytes that are read exactly once per token: bypass L1 (".nc" read-only path, no allocate).
__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// Asynchronous 16-byte global -> shared copy (L2 only). Many can be in flight per thread without
// holding registers; cp_async_wait_all() waits for this thread's copies.
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }

// Tensor-core multiply-accumulate on FP8: D[16x8] += A[16x32] * B[32x8], fp32 accumulators.
// We use it as a dot-product engine: A holds the (quantized) activation vector in every row,
// B holds 8 weight rows. Each thread passes 4 registers of A, 2 of B and gets 4 outputs back.
__device__ __forceinline__ void mma_fp8(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// The reference quantizer is a Triton kernel, and Triton compiles fp32 `/` to `div.full.f32`
// (an approximate division, up to 2 ulp off), not to IEEE `div.rn.f32`. Because x and max|x| are
// both bf16, x/s very often lands exactly on a rounding tie between two fp8 values, and a 1-ulp
// difference in the quotient then picks the other fp8 value. Using the same instruction makes the
// quantized activations bit-identical to the reference.
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// Quantize 4 floats to 4 packed fp8 e4m3 bytes (round-to-nearest-even, saturate to +-448).
__device__ __forceinline__ uint32_t quant4(float a, float b, float c, float d, float s) {
  float inv_a = s > 0.f ? div_full(a, s) : 0.f, inv_b = s > 0.f ? div_full(b, s) : 0.f;
  float inv_c = s > 0.f ? div_full(c, s) : 0.f, inv_d = s > 0.f ? div_full(d, s) : 0.f;
  __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(inv_a, inv_b), __NV_SATFINITE, __NV_E4M3);
  __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(inv_c, inv_d), __NV_SATFINITE, __NV_E4M3);
  return (uint32_t)lo | ((uint32_t)hi << 16);
}

// ------------------------------------------------------------------------------------------------
// FP8 block-scaled GEMV:  y[N] = W[N, K] @ x[K]   (one token)
//
// Weight layout ("mma-tiled", built once at load time by model.py): rows are taken in groups of
// 8, K in chunks of 64. Each 8x64 = 512-byte tile is stored as 32 slices of 16 bytes, and slice L
// is exactly what lane L needs as its two mma B-fragments for two consecutive k32 steps. So the
// whole hot loop is: one coalesced 16-byte load per lane -> two mma instructions. No fp8->float
// conversions, no shuffles, no shared-memory traffic for weights.
//
// Prologue (per CTA, overlapped with the first weight loads): stage x in shared memory, optionally
// apply RMSNorm, then quantize x to fp8 per 128-group exactly like the reference.
// Epilogue: plain store, residual add (in place), or SwiGLU (silu(gate) * up).
// ------------------------------------------------------------------------------------------------
enum { PRO_QUANT = 0, PRO_RMSNORM_QUANT = 1 };
enum { EPI_STORE = 0, EPI_RESIDUAL = 1, EPI_SWIGLU = 2 };

template <int K, int WARPS, int PRO, int EPI, int DEPTH>
__global__ void __launch_bounds__(WARPS * 32)
fp8_gemv_kernel(const uint4* __restrict__ w,            // mma-tiled fp8 weights, N*K bytes
                const float* __restrict__ w_scale,      // [N/128, K/128]
                const bf16* __restrict__ x,             // [K] input (the residual stream for RMSNorm)
                const bf16* __restrict__ norm_w,        // [K] RMSNorm weight (PRO_RMSNORM_QUANT only)
                bf16* __restrict__ out,                 // STORE: [N]; RESIDUAL: residual [N] (in place); SWIGLU: [N/2]
                int N, float eps,
                int pre,    // pipeline slots filled before waiting on the previous kernel (rest right after)
                int trig,   // K block at which the next kernel may launch (-1: right after our wait)
                int kmajor) {  // weight tile order: 0 = all K tiles of a row group together, 1 = K-major
  constexpr int KB = K / QBLOCK;                 // 128-wide K blocks
  constexpr int NG = (EPI == EPI_SWIGLU) ? 2 : 1;  // 8-row groups per warp (gate + up for SwiGLU)
  constexpr int TILES_PER_GROUP = K / 64;        // 512-byte tiles per 8-row group
  static_assert(KB % DEPTH == 0, "pipeline depth must divide the number of K blocks");

  __shared__ __align__(16) bf16 xs_bf[K];        // raw input, bf16
  __shared__ __align__(16) uint32_t xq[K / 4];   // quantized input, 4 fp8 per word
  __shared__ float xscale[KB];                   // activation scale per 128-block
  __shared__ float red[WARPS];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int t = lane & 3;                        // "thread in group" in mma fragment terms
  const int task = blockIdx.x * WARPS + warp;    // which 8-row output group this warp owns
  // Timeline region per GEMV kind (qkv, o_proj, gate_up, down), 4 stamps per warp.
  unsigned long long* tl = g_timeline;
  if (tl) tl += (size_t)(N == 6144 ? 0 : K == 12288 ? 3 : N == 24576 ? 2 : 1) * 4 * 4096;
  if (tl && lane == 0) tl[task * 4 + 0] = globaltimer();
  const int n_tasks = N / (8 * NG);
  const bool active = task < n_tasks;

  // ---- 1. Start streaming weights immediately: they do not depend on x. ----
  int grp[NG];
  grp[0] = task;
  if (NG == 2) grp[NG - 1] = task + N / 16;      // the matching "up" rows live in the second half
  // Address of tile t of row group grp (in uint4 = 16-byte units): group-major puts a group's K tiles
  // next to each other; K-major puts tile t of all groups next to each other, so at any moment the
  // warps of the whole grid read one contiguous window (better DRAM page locality).
  const size_t tstride = kmajor ? (size_t)(N / 8) * 32 : 32;
  const uint4* wp[NG];
#pragma unroll
  for (int g = 0; g < NG; ++g) {
    const size_t gi = active ? grp[g] : 0;
    wp[g] = w + (kmajor ? gi * 32 : gi * TILES_PER_GROUP * 32) + lane;
  }

  uint4 buf[DEPTH][NG][2];
  auto prefetch = [&](bool before_wait) {
#pragma unroll
    for (int s = 0; s < DEPTH; ++s)
      if ((s < pre) == before_wait)
#pragma unroll
        for (int g = 0; g < NG; ++g) {
          buf[s][g][0] = ld_stream(wp[g] + (2 * s) * tstride);
          buf[s][g][1] = ld_stream(wp[g] + (2 * s + 1) * tstride);
        }
  };
  prefetch(true);

  // The RMSNorm weight is a constant too: fetch this lane's slice before waiting.
  constexpr int GPW = KB / WARPS;            // 128-groups this warp quantizes
  constexpr int CH = GPW % 8 == 0 ? 8 : (GPW % 4 == 0 ? 4 : (GPW % 2 == 0 ? 2 : 1));  // groups reduced together (ILP)
  static_assert(GPW % CH == 0, "groups per warp must be a multiple of the chunk");
  uint2 nw[PRO == PRO_RMSNORM_QUANT ? GPW : 1];
  if (PRO == PRO_RMSNORM_QUANT) {
#pragma unroll
    for (int i = 0; i < GPW; ++i)
      nw[i] = *reinterpret_cast<const uint2*>(norm_w + (warp + i * WARPS) * QBLOCK + lane * 4);
  }

  pdl_wait();      // x / the residual stream come from the previous kernel
  if (tl && lane == 0) tl[task * 4 + 1] = globaltimer();
  if (trig < 0) pdl_trigger();  // let the next kernel launch now (it only prefetches, then waits)

  // ---- 2. Prologue: x -> (RMSNorm) -> fp8 per 128-group, into shared memory. ----
  {
    // Stage x with async copies: every copy is in flight at once, so this costs one L2 round trip.
    // The rest of the weight prefetch goes first: DRAM time is the bottleneck, so the weight requests
    // must not wait behind the x copies (measured: issuing x first shortens the prologue but costs
    // 0.6% end to end, because the prefetch already hides the prologue).
    prefetch(false);
    for (int i = threadIdx.x; i < K / 8; i += WARPS * 32) cp_async16(xs_bf + i * 8, x + i * 8);
    cp_async_wait_all();
    __syncthreads();
    float rstd = 0.f;
    if (PRO == PRO_RMSNORM_QUANT) {
      float ss = 0.f;
      for (int i = threadIdx.x; i < K / 8; i += WARPS * 32) {
        const uint4 v = reinterpret_cast<const uint4*>(xs_bf)[i];
        const bf16* e = reinterpret_cast<const bf16*>(&v);
#pragma unroll
        for (int j = 0; j < 8; ++j) { float f = __bfloat162float(e[j]); ss += f * f; }
      }
      ss = warp_sum(ss);
      if (lane == 0) red[warp] = ss;
      __syncthreads();
      float tot = 0.f;
#pragma unroll
      for (int i = 0; i < WARPS; ++i) tot += red[i];
      rstd = rsqrtf(tot / (float)K + eps);
    }
    // Quantize: warp w owns groups w, w+WARPS, ...; lane holds 4 consecutive values of each.
    // CH groups at a time so their max-reductions (shuffles) overlap instead of running back to back.
#pragma unroll
    for (int c0 = 0; c0 < GPW; c0 += CH) {
      float v[CH][4], amax[CH];
#pragma unroll
      for (int c = 0; c < CH; ++c) {
        const int k0 = (warp + (c0 + c) * WARPS) * QBLOCK + lane * 4;
        const uint2 raw = *reinterpret_cast<const uint2*>(xs_bf + k0);
        const bf16* e = reinterpret_cast<const bf16*>(&raw);
        const bf16* we = reinterpret_cast<const bf16*>(&nw[PRO == PRO_RMSNORM_QUANT ? c0 + c : 0]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float f = __bfloat162float(e[j]);
          // RMSNorm as the reference: normalize in fp32, round to bf16, multiply by weight in bf16.
          if (PRO == PRO_RMSNORM_QUANT) f = bf16r(__bfloat162float(we[j]) * bf16r(f * rstd));
          v[c][j] = f;
        }
        amax[c] = fmaxf(fmaxf(fabsf(v[c][0]), fabsf(v[c][1])), fmaxf(fabsf(v[c][2]), fabsf(v[c][3])));
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1)
#pragma unroll
        for (int c = 0; c < CH; ++c) amax[c] = fmaxf(amax[c], __shfl_xor_sync(0xffffffffu, amax[c], o));
#pragma unroll
      for (int c = 0; c < CH; ++c) {
        const int blk = warp + (c0 + c) * WARPS;
        const float sc = div_full(amax[c], 448.0f);  // same instruction as the reference
        xq[(blk * QBLOCK + lane * 4) / 4] = quant4(v[c][0], v[c][1], v[c][2], v[c][3], sc);
        if (lane == 0) xscale[blk] = sc;
      }
    }
  }
  __syncthreads();
  if (tl && lane == 0) tl[task * 4 + 2] = globaltimer();
  if (!active) return;

  // ---- 3. Main loop: per 128-wide K block, 4 mma steps of k=32, then scale and accumulate. ----
  float acc[NG][2];
#pragma unroll
  for (int g = 0; g < NG; ++g) acc[g][0] = acc[g][1] = 0.f;
  const float* sw_row[NG];
#pragma unroll
  for (int g = 0; g < NG; ++g) sw_row[g] = w_scale + (size_t)(grp[g] / 16) * KB;  // 16 groups of 8 rows = 128 rows

  for (int kb0 = 0; kb0 < KB; kb0 += DEPTH) {
#pragma unroll
    for (int s = 0; s < DEPTH; ++s) {
      const int kb = kb0 + s;
      if (kb == trig) pdl_trigger();  // launch the next kernel near our tail, so its prefetch fills our drain
      uint4 cur[NG][2];
#pragma unroll
      for (int g = 0; g < NG; ++g) { cur[g][0] = buf[s][g][0]; cur[g][1] = buf[s][g][1]; }
      if (kb + DEPTH < KB) {  // refill this pipeline slot with the block DEPTH steps ahead
#pragma unroll
        for (int g = 0; g < NG; ++g) {
          buf[s][g][0] = ld_stream(wp[g] + (2 * (kb + DEPTH)) * tstride);
          buf[s][g][1] = ld_stream(wp[g] + (2 * (kb + DEPTH) + 1) * tstride);
        }
      }
      // A fragments for the 4 k32 chunks of this block: rows of A are all equal to x.
      // Chunk c needs bytes [4t, 4t+4) and [16+4t, 16+4t+4) of the chunk -> words c*8+t and c*8+4+t.
      uint32_t alo[4], ahi[4];
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        alo[c] = xq[(kb * 4 + c) * 8 + t];
        ahi[c] = xq[(kb * 4 + c) * 8 + 4 + t];
      }
      const float sx = xscale[kb];
#pragma unroll
      for (int g = 0; g < NG; ++g) {
        float d[4] = {0.f, 0.f, 0.f, 0.f};
        mma_fp8(d, alo[0], alo[0], ahi[0], ahi[0], cur[g][0].x, cur[g][0].y);
        mma_fp8(d, alo[1], alo[1], ahi[1], ahi[1], cur[g][0].z, cur[g][0].w);
        mma_fp8(d, alo[2], alo[2], ahi[2], ahi[2], cur[g][1].x, cur[g][1].y);
        mma_fp8(d, alo[3], alo[3], ahi[3], ahi[3], cur[g][1].z, cur[g][1].w);
        const float sw = __ldg(sw_row[g] + kb);
        acc[g][0] += d[0] * sx * sw;   // d[0], d[1] = outputs for weight rows 2t and 2t+1 of the group
        acc[g][1] += d[1] * sx * sw;
      }
    }
  }

  if (tl && lane == 0) tl[task * 4 + 3] = globaltimer();
  // ---- 4. Epilogue. Lanes 0..3 hold rows 2t, 2t+1 (all 8 row-groups of lanes hold copies). ----
  if (lane >= 4) return;
  const int n = grp[0] * 8 + 2 * t;
  if (EPI == EPI_STORE) {
    reinterpret_cast<__nv_bfloat162*>(out)[n / 2] = __floats2bfloat162_rn(acc[0][0], acc[0][1]);
  } else if (EPI == EPI_RESIDUAL) {
    __nv_bfloat162 r = reinterpret_cast<__nv_bfloat162*>(out)[n / 2];
    float y0 = bf16r(acc[0][0]), y1 = bf16r(acc[0][1]);
    reinterpret_cast<__nv_bfloat162*>(out)[n / 2] =
        __floats2bfloat162_rn(__low2float(r) + y0, __high2float(r) + y1);
  } else {  // SwiGLU: h = silu(gate) * up, each op rounded to bf16 like the reference
    float h[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float gt = bf16r(acc[0][j]);
      float up = bf16r(acc[NG - 1][j]);
      float act = bf16r(gt / (1.0f + expf(-gt)));
      h[j] = act * up;
    }
    reinterpret_cast<__nv_bfloat162*>(out)[n / 2] = __floats2bfloat162_rn(h[0], h[1]);
  }
}

// ------------------------------------------------------------------------------------------------
// Per-head RMSNorm + RoPE for one 128-dim head, one warp: lane owns dims [4*lane, 4*lane+4).
// rotate_half pairs dim d with d+64, i.e. lane l with lane l^16. Every step rounds to bf16 like the
// reference (normalize in fp32 -> bf16 -> * weight -> bf16; x*cos and rot*sin rounded separately).
// The caller rounds the result to bf16 when storing it.
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ void norm_rope_head(const bf16* __restrict__ src, const bf16* __restrict__ nw,
                                               const bf16* __restrict__ cos_row, const bf16* __restrict__ sin_row,
                                               int lane, float eps, float (&o)[4]) {
  const int d0 = lane * 4;
  float v[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(src[d0 + j]);
  const float ss = warp_sum(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
  const float rstd = rsqrtf(ss / (float)HEAD_DIM + eps);
#pragma unroll
  for (int j = 0; j < 4; ++j) v[j] = bf16r(__bfloat162float(nw[d0 + j]) * bf16r(v[j] * rstd));
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float partner = __shfl_xor_sync(0xffffffffu, v[j], 16);
    const float rot = (lane < 16) ? -partner : partner;
    o[j] = bf16r(v[j] * __bfloat162float(cos_row[d0 + j])) + bf16r(rot * __bfloat162float(sin_row[d0 + j]));
  }
}

// ------------------------------------------------------------------------------------------------
// Decode attention (GQA, split along the KV length = "flash-decoding"), with the q/k RMSNorm,
// RoPE and the KV-cache write of the new token fused in.
//
// Grid: (kv_head, split). Each CTA handles one KV head (and its 4 query heads) over a CHUNK-key chunk.
// 1. Before waiting on the previous kernel it already issues the loads of the cached keys/values:
//    they were written in earlier steps. Only q and the new token's k/v come from the qkv GEMV.
// 2. Each CTA normalizes + rotates its 4 query heads itself (cheap, 4 x 128 values). The CTA whose
//    chunk contains the new position also normalizes + rotates the new key, writes k and v into the
//    cache and uses them directly.
// 3. Scores (128/CHUNK threads per key), softmax over the chunk, P @ V, unnormalized partial result.
// 4. The last CTA to finish for a KV head (atomic counter) merges all partials, so no separate
//    combine kernel is launched. The grid is sized for the maximum context; CTAs past the current
//    length exit at once, which keeps the launch shape static for CUDA-graph replay.
// ------------------------------------------------------------------------------------------------
constexpr int MAX_SPLITS = 128;  // splits of the smallest chunk (32 keys) => contexts up to 4096

template <int CHUNK>  // keys per CTA: 32, 64 or 128
__global__ void __launch_bounds__(128)
attention_kernel(const bf16* __restrict__ qkv,                 // [6144] raw q | k | v of this token
                 const bf16* __restrict__ q_norm_w, const bf16* __restrict__ k_norm_w,  // [128]
                 const bf16* __restrict__ cos_tab, const bf16* __restrict__ sin_tab,    // [max_seq, 128]
                 const int* __restrict__ state,                // [pos, step]
                 bf16* __restrict__ k_cache, bf16* __restrict__ v_cache,  // [8, max_seq, 128], this layer
                 float* __restrict__ o_part,                   // [32, max_splits, 128]
                 float* __restrict__ ml_part,                  // [32, max_splits, 2]
                 int* __restrict__ counters,                   // [8], zero between launches
                 bf16* __restrict__ out,                       // [32*128]
                 int max_seq, int max_splits, float scale, float eps,
                 int late_trigger) {  // 1: let the next kernel launch only after our partial results are out
  constexpr int TPK = 128 / CHUNK;       // threads per key
  constexpr int DPT = HEAD_DIM / TPK;    // head dims per thread in the score dot product
  constexpr int PER_LANE = CHUNK / 32;   // softmax values per lane
  const int kvh = blockIdx.x, split = blockIdx.y;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  unsigned long long* tl = g_timeline ? g_timeline + 4 * 4 * 4096 + (size_t)(kvh * MAX_SPLITS + split) * 8 : nullptr;
  if (tl && tid == 0) tl[0] = globaltimer();
  // The position is written by the previous step's last kernel, so it can be read before pdl_wait.
  const int pos = state[0];
  const int ctx = pos + 1;
  const int n_splits = (ctx + CHUNK - 1) / CHUNK;
  if (split >= n_splits) return;
  const int j0 = split * CHUNK;
  const int nj = min(CHUNK, ctx - j0);
  const bool owns_new = (split == n_splits - 1);  // the new token's row is in this chunk
  const int jn = pos - j0;                        // ...at this row (valid if owns_new)

  __shared__ float qs[GQA][HEAD_DIM];
  __shared__ float sc[GQA][CHUNK];
  __shared__ float part[TPK > 1 ? TPK - 1 : 1][GQA][CHUNK];
  __shared__ __align__(16) bf16 vs[CHUNK][HEAD_DIM];
  __shared__ __align__(16) bf16 knew[HEAD_DIM];
  __shared__ float m_s[GQA], l_s[GQA];
  __shared__ bool is_last;

  // ---- 1. Cached keys (into registers) and values (into shared memory), before waiting. ----
  const bf16* kb = k_cache + ((size_t)kvh * max_seq + j0) * HEAD_DIM;
  const bf16* vb = v_cache + ((size_t)kvh * max_seq + j0) * HEAD_DIM;
  const int key = tid % CHUNK, slice = tid / CHUNK;  // TPK threads per key, DPT dims each
  const bool is_new = owns_new && key == jn;
  uint4 kr[DPT / 8];
  if (key < nj && !is_new) {
#pragma unroll
    for (int i = 0; i < DPT / 8; ++i)
      kr[i] = *reinterpret_cast<const uint4*>(kb + key * HEAD_DIM + slice * DPT + i * 8);
  }
  for (int i = tid; i < nj * (HEAD_DIM / 8); i += 128) {
    const int row = i >> 4;
    if (!(owns_new && row == jn)) cp_async16(&vs[row][(i & 15) * 8], vb + row * HEAD_DIM + (i & 15) * 8);
  }

  pdl_wait();      // q, k, v of the new token come from the qkv GEMV
  if (tl && tid == 0) tl[1] = globaltimer();
  if (!late_trigger) pdl_trigger();

  // ---- 2. q (and the new k, v) normalization + RoPE. ----
  const bf16* cos_row = cos_tab + (size_t)pos * HEAD_DIM;
  const bf16* sin_row = sin_tab + (size_t)pos * HEAD_DIM;
  {
    float o[4];
    norm_rope_head(qkv + (kvh * GQA + warp) * HEAD_DIM, q_norm_w, cos_row, sin_row, lane, eps, o);
#pragma unroll
    for (int j = 0; j < 4; ++j) qs[warp][lane * 4 + j] = bf16r(o[j]);
  }
  if (owns_new && warp == 0) {
    float o[4];
    norm_rope_head(qkv + (N_Q_HEADS + kvh) * HEAD_DIM, k_norm_w, cos_row, sin_row, lane, eps, o);
    bf16* dst = k_cache + ((size_t)kvh * max_seq + pos) * HEAD_DIM + lane * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) { dst[j] = __float2bfloat16_rn(o[j]); knew[lane * 4 + j] = __float2bfloat16_rn(o[j]); }
  } else if (owns_new && warp == 1) {
    const bf16* src = qkv + (N_Q_HEADS + N_KV_HEADS + kvh) * HEAD_DIM + lane * 4;
    bf16* dst = v_cache + ((size_t)kvh * max_seq + pos) * HEAD_DIM + lane * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) { dst[j] = src[j]; vs[jn][lane * 4 + j] = src[j]; }
  }
  cp_async_wait_all();
  __syncthreads();
  if (tl && tid == 0) tl[2] = globaltimer();

  // ---- 3. Scores: thread (key, slice) does DPT of the 128 dims for all 4 query heads. ----
  float dot[GQA] = {0.f, 0.f, 0.f, 0.f};
  if (key < nj) {
    if (is_new) {
#pragma unroll
      for (int i = 0; i < DPT / 8; ++i) kr[i] = *reinterpret_cast<const uint4*>(knew + slice * DPT + i * 8);
    }
#pragma unroll
    for (int i = 0; i < DPT / 8; ++i) {
      const bf16* e = reinterpret_cast<const bf16*>(&kr[i]);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float kf = __bfloat162float(e[j]);
#pragma unroll
        for (int h = 0; h < GQA; ++h) dot[h] += qs[h][slice * DPT + i * 8 + j] * kf;
      }
    }
  }
  if (TPK > 1 && slice > 0) {
#pragma unroll
    for (int h = 0; h < GQA; ++h) part[slice - 1][h][key] = dot[h];
  }
  if (TPK > 1) __syncthreads();
  if (slice == 0) {
#pragma unroll
    for (int h = 0; h < GQA; ++h) {
      float t = dot[h];
#pragma unroll
      for (int p = 0; p + 1 < TPK; ++p) t += part[p][h][key];
      sc[h][key] = (key < nj) ? t * scale : -INFINITY;
    }
  }
  __syncthreads();
  if (tl && tid == 0) tl[3] = globaltimer();

  {  // softmax over the chunk, one warp per query head
    float p[PER_LANE];
    float m = -INFINITY;
#pragma unroll
    for (int i = 0; i < PER_LANE; ++i) { p[i] = sc[warp][lane + 32 * i]; m = fmaxf(m, p[i]); }
    m = warp_max(m);
    float l = 0.f;
#pragma unroll
    for (int i = 0; i < PER_LANE; ++i) { p[i] = expf(p[i] - m); l += p[i]; sc[warp][lane + 32 * i] = p[i]; }
    l = warp_sum(l);
    if (lane == 0) { m_s[warp] = m; l_s[warp] = l; }
  }
  __syncthreads();

  // ---- P @ V: thread d owns output dim d for all 4 heads. ----
  const int d = tid;
  float acc[GQA] = {0.f, 0.f, 0.f, 0.f};
  for (int j = 0; j < nj; ++j) {
    const float vf = __bfloat162float(vs[j][d]);
#pragma unroll
    for (int h = 0; h < GQA; ++h) acc[h] += sc[h][j] * vf;
  }
#pragma unroll
  for (int h = 0; h < GQA; ++h) {
    const int qh = kvh * GQA + h;
    o_part[((size_t)qh * max_splits + split) * HEAD_DIM + d] = acc[h];
    if (tid == 0) {
      ml_part[((size_t)qh * max_splits + split) * 2 + 0] = m_s[h];
      ml_part[((size_t)qh * max_splits + split) * 2 + 1] = l_s[h];
    }
  }

  // ---- 4. The last CTA for this KV head merges all splits. ----
  __threadfence();
  __syncthreads();
  if (tl && tid == 0) tl[4] = globaltimer();
  if (late_trigger) pdl_trigger();
  if (tid == 0) is_last = (atomicAdd(&counters[kvh], 1) == n_splits - 1);
  __syncthreads();
  if (tl && tid == 0) tl[5] = globaltimer();
  if (!is_last) return;
  __threadfence();
  // One warp per query head: global max of the split maxima, per-split weights exp(m_s - M) (kept in
  // shared memory) and the total softmax denominator L.
  __shared__ float cw[GQA][MAX_SPLITS];
  __shared__ float cl[GQA];
  {
    const int qh = kvh * GQA + warp;
    const float* ml = ml_part + (size_t)qh * max_splits * 2;
    float mloc = -INFINITY;
    for (int s = lane; s < n_splits; s += 32) mloc = fmaxf(mloc, __ldcg(ml + 2 * s));
    const float M = warp_max(mloc);
    float lsum = 0.f;
    for (int s = lane; s < n_splits; s += 32) {
      const float wgt = expf(__ldcg(ml + 2 * s) - M);
      cw[warp][s] = wgt;
      lsum += wgt * __ldcg(ml + 2 * s + 1);
    }
    lsum = warp_sum(lsum);
    if (lane == 0) cl[warp] = lsum;
  }
  __syncthreads();
  float o[GQA] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 8  // 8 splits x 4 heads = 32 independent L2 loads in flight per thread
  for (int s = 0; s < n_splits; ++s) {
#pragma unroll
    for (int h = 0; h < GQA; ++h)
      o[h] += cw[h][s] * __ldcg(o_part + ((size_t)(kvh * GQA + h) * max_splits + s) * HEAD_DIM + d);
  }
#pragma unroll
  for (int h = 0; h < GQA; ++h) out[(kvh * GQA + h) * HEAD_DIM + d] = __float2bfloat16_rn(o[h] / cl[h]);
  if (tid == 0) counters[kvh] = 0;
  if (tl && tid == 0) tl[6] = globaltimer();
}

// ------------------------------------------------------------------------------------------------
// lm_head (bf16 GEMV, 151936 x 4096) with the final RMSNorm as prologue and greedy argmax fused in.
// Persistent CTAs: the prologue (norm of the 4096-vector) is computed once per CTA, then each warp
// walks over rows. Argmax: each CTA folds its best (logit, index) into one 64-bit atomicMax.
// Ties go to the smallest index, like torch.argmax.
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ unsigned long long pack_argmax(float v, int idx) {
  uint32_t u = __float_as_uint(v);
  u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);  // order-preserving float -> uint
  return ((unsigned long long)u << 32) | (uint32_t)(0xFFFFFFFFu - (uint32_t)idx);
}

template <int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
lm_head_argmax_kernel(const uint4* __restrict__ w,          // [V, 4096] bf16 row-major
                      const bf16* __restrict__ x,           // [4096] residual stream
                      const bf16* __restrict__ norm_w,      // [4096] final norm weight
                      bf16* __restrict__ logits,            // [V]
                      unsigned long long* __restrict__ best,  // packed (value, index), zero before launch
                      int V, float eps) {
  constexpr int K = HIDDEN;
  constexpr int LOADS = K / 8 / 32;  // uint4 (8 bf16) loads per lane per row = 16
  __shared__ __align__(16) float xn[K];
  __shared__ float red[WARPS];
  __shared__ unsigned long long cta_best[WARPS];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int total_warps = gridDim.x * WARPS;
  int row = blockIdx.x * WARPS + warp;

  // Prefetch this warp's first row before the prologue.
  uint4 cur[LOADS];
  if (row < V) {
#pragma unroll
    for (int i = 0; i < LOADS; ++i) cur[i] = ld_stream(w + (size_t)row * (K / 8) + i * 32 + lane);
  }

  pdl_wait();
  pdl_trigger();
  float ss = 0.f;
  for (int i = threadIdx.x; i < K; i += WARPS * 32) { float f = __bfloat162float(x[i]); xn[i] = f; ss += f * f; }
  ss = warp_sum(ss);
  if (lane == 0) red[warp] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < WARPS; ++i) tot += red[i];
  const float rstd = rsqrtf(tot / (float)K + eps);
  __syncthreads();
  for (int i = threadIdx.x; i < K; i += WARPS * 32) xn[i] = bf16r(__bfloat162float(norm_w[i]) * bf16r(xn[i] * rstd));
  __syncthreads();

  float best_v = -INFINITY;
  int best_i = 0x7FFFFFFF;
  while (row < V) {
    const int next = row + total_warps;
    uint4 nxt[LOADS];
    if (next < V) {
#pragma unroll
      for (int i = 0; i < LOADS; ++i) nxt[i] = ld_stream(w + (size_t)next * (K / 8) + i * 32 + lane);
    }
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < LOADS; ++i) {
      const int k0 = (i * 32 + lane) * 8;
      const uint32_t* p = reinterpret_cast<const uint32_t*>(&cur[i]);
      const float4 x0 = *reinterpret_cast<const float4*>(&xn[k0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&xn[k0 + 4]);
      // bf16 -> fp32 is a 16-bit shift: low half << 16, high half masked.
      acc += __uint_as_float(p[0] << 16) * x0.x + __uint_as_float(p[0] & 0xFFFF0000u) * x0.y;
      acc += __uint_as_float(p[1] << 16) * x0.z + __uint_as_float(p[1] & 0xFFFF0000u) * x0.w;
      acc += __uint_as_float(p[2] << 16) * x1.x + __uint_as_float(p[2] & 0xFFFF0000u) * x1.y;
      acc += __uint_as_float(p[3] << 16) * x1.z + __uint_as_float(p[3] & 0xFFFF0000u) * x1.w;
    }
    acc = warp_sum(acc);
    const float lv = bf16r(acc);  // logits are bf16 in the reference
    if (lane == 0) logits[row] = __float2bfloat16_rn(acc);
    if (lv > best_v || (lv == best_v && row < best_i)) { best_v = lv; best_i = row; }
    row = next;
#pragma unroll
    for (int i = 0; i < LOADS; ++i) cur[i] = nxt[i];
  }
  if (lane == 0) cta_best[warp] = (best_i == 0x7FFFFFFF) ? 0ull : pack_argmax(best_v, best_i);
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned long long b = 0;
#pragma unroll
    for (int i = 0; i < WARPS; ++i) b = cta_best[i] > b ? cta_best[i] : b;
    atomicMax(best, b);
  }
}

// ------------------------------------------------------------------------------------------------
// Step bookkeeping kernels.
// ------------------------------------------------------------------------------------------------
// resid = embed[token]
__global__ void embed_kernel(const int* __restrict__ token, const bf16* __restrict__ table, bf16* __restrict__ resid) {
  pdl_wait();
  pdl_trigger();
  const int tok = *token;
  const uint4* src = reinterpret_cast<const uint4*>(table + (size_t)tok * HIDDEN);
  reinterpret_cast<uint4*>(resid)[threadIdx.x] = src[threadIdx.x];  // 512 threads x 8 bf16
}

// Decode the argmax, record the token, advance the position. state = [pos, step].
__global__ void finalize_kernel(unsigned long long* __restrict__ best, int* __restrict__ token,
                                int* __restrict__ out_tokens, int* __restrict__ state) {
  pdl_wait();
  pdl_trigger();
  const unsigned long long b = *best;
  const int tok = (int)(0xFFFFFFFFu - (uint32_t)(b & 0xFFFFFFFFull));
  *token = tok;
  out_tokens[state[1]] = tok;
  state[0] += 1;
  state[1] += 1;
  *best = 0ull;
}

}  // namespace

// ================================================================================================
// Host-side launchers (called from Python, captured into a CUDA graph).
// ================================================================================================
#define CHECK_LAUNCH() C10_CUDA_KERNEL_LAUNCH_CHECK()

static bool g_pdl = false;  // launch with programmatic dependent launch (set from Python)

template <typename... KArgs, typename... Args>
static void launch(void (*kernel)(KArgs...), dim3 grid, dim3 block, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = g_pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...));
}

void set_pdl(bool on) { g_pdl = on; }

// Point the GEMV timestamp buffer at `buf` (uint64, >= 4 per warp of the largest grid), or disable.
void set_timeline(c10::optional<torch::Tensor> buf) {
  unsigned long long* p = buf.has_value() ? reinterpret_cast<unsigned long long*>(buf->data_ptr()) : nullptr;
  C10_CUDA_CHECK(cudaMemcpyToSymbol(g_timeline, &p, sizeof(p)));
}

template <int K, int WARPS, int PRO, int EPI, int DEPTH>
static void launch_gemv(const torch::Tensor& w, const torch::Tensor& ws, const torch::Tensor& x,
                        const c10::optional<torch::Tensor>& norm_w, torch::Tensor& out, int N, double eps,
                        int pre = DEPTH, int trig = -1, int kmajor = 0) {
  const int n_tasks = (EPI == EPI_SWIGLU) ? N / 16 : N / 8;
  const int grid = (n_tasks + WARPS - 1) / WARPS;
  launch(fp8_gemv_kernel<K, WARPS, PRO, EPI, DEPTH>, grid, WARPS * 32,
         reinterpret_cast<const uint4*>(w.data_ptr()), (const float*)ws.data_ptr<float>(),
         reinterpret_cast<const bf16*>(x.data_ptr()),
         norm_w.has_value() ? reinterpret_cast<const bf16*>(norm_w->data_ptr()) : (const bf16*)nullptr,
         reinterpret_cast<bf16*>(out.data_ptr()), N, (float)eps, pre, trig, kmajor);
}

// kind: 0 = qkv (rmsnorm, store), 1 = o_proj (quant, residual), 2 = gate_up (rmsnorm, swiglu), 3 = down (quant, residual)
// depth: 128-wide K blocks each warp keeps in flight (register pipeline depth); warps: warps per CTA.
template <int DEPTH, int WARPS>
static void gemv_kind(int64_t kind, torch::Tensor& w, torch::Tensor& ws, torch::Tensor& x,
                      c10::optional<torch::Tensor>& norm_w, torch::Tensor& out, int64_t N, double eps, int pre,
                      int trig, int km) {
  switch (kind) {
    case 0: launch_gemv<4096, WARPS, PRO_RMSNORM_QUANT, EPI_STORE, DEPTH>(w, ws, x, norm_w, out, N, eps, pre, trig, km); break;
    case 1: launch_gemv<4096, WARPS, PRO_QUANT, EPI_RESIDUAL, DEPTH>(w, ws, x, norm_w, out, N, eps, pre, trig, km); break;
    case 2: launch_gemv<4096, WARPS, PRO_RMSNORM_QUANT, EPI_SWIGLU, DEPTH>(w, ws, x, norm_w, out, N, eps, pre, trig, km); break;
    case 3: launch_gemv<12288, WARPS, PRO_QUANT, EPI_RESIDUAL, DEPTH>(w, ws, x, norm_w, out, N, eps, pre, trig, km); break;
    default: TORCH_CHECK(false, "unknown gemv kind");
  }
}

template <int DEPTH>
static void gemv_warps(int64_t kind, int64_t warps, torch::Tensor& w, torch::Tensor& ws, torch::Tensor& x,
                       c10::optional<torch::Tensor>& norm_w, torch::Tensor& out, int64_t N, double eps, int pre,
                       int trig, int km) {
  switch (warps) {
    case 1: gemv_kind<DEPTH, 1>(kind, w, ws, x, norm_w, out, N, eps, pre, trig, km); break;
    case 2: gemv_kind<DEPTH, 2>(kind, w, ws, x, norm_w, out, N, eps, pre, trig, km); break;
    case 4: gemv_kind<DEPTH, 4>(kind, w, ws, x, norm_w, out, N, eps, pre, trig, km); break;
    case 8: gemv_kind<DEPTH, 8>(kind, w, ws, x, norm_w, out, N, eps, pre, trig, km); break;
    default: TORCH_CHECK(false, "warps must be 1, 2, 4 or 8");
  }
}

// pre: pipeline slots prefetched before waiting on the previous kernel (default: all);
// trig: K block at which the next kernel is allowed to launch (-1: right after the wait);
// kmajor: weight tile order (see model.tile_fp8).
void fp8_gemv(int64_t kind, torch::Tensor w, torch::Tensor ws, torch::Tensor x,
              c10::optional<torch::Tensor> norm_w, torch::Tensor out, int64_t N, double eps, int64_t depth,
              int64_t warps, int64_t pre, int64_t trig, int64_t kmajor) {
  if (pre < 0) pre = depth;
  switch (depth) {
    case 4: gemv_warps<4>(kind, warps, w, ws, x, norm_w, out, N, eps, (int)pre, (int)trig, (int)kmajor); break;
    case 8: gemv_warps<8>(kind, warps, w, ws, x, norm_w, out, N, eps, (int)pre, (int)trig, (int)kmajor); break;
    default: TORCH_CHECK(false, "depth must be 4 or 8");
  }
}

void attention(torch::Tensor qkv, torch::Tensor q_norm_w, torch::Tensor k_norm_w, torch::Tensor cos_tab,
               torch::Tensor sin_tab, torch::Tensor state, torch::Tensor k_cache, torch::Tensor v_cache,
               torch::Tensor o_part, torch::Tensor ml_part, torch::Tensor counters, torch::Tensor out, double scale,
               double eps, int64_t late_trigger, int64_t chunk) {
  const int max_seq = k_cache.size(1);
  const int max_splits = o_part.size(1);
  TORCH_CHECK(max_splits * chunk >= max_seq, "o_part too small for max_seq");
  TORCH_CHECK((max_seq + chunk - 1) / chunk <= MAX_SPLITS, "max_seq too large for MAX_SPLITS");
  auto run = [&](auto kernel, int c) {
    launch(kernel, dim3(N_KV_HEADS, (max_seq + c - 1) / c), 128,
           reinterpret_cast<const bf16*>(qkv.data_ptr()), reinterpret_cast<const bf16*>(q_norm_w.data_ptr()),
           reinterpret_cast<const bf16*>(k_norm_w.data_ptr()), reinterpret_cast<const bf16*>(cos_tab.data_ptr()),
           reinterpret_cast<const bf16*>(sin_tab.data_ptr()), (const int*)state.data_ptr<int>(),
           reinterpret_cast<bf16*>(k_cache.data_ptr()), reinterpret_cast<bf16*>(v_cache.data_ptr()),
           o_part.data_ptr<float>(), ml_part.data_ptr<float>(), counters.data_ptr<int>(),
           reinterpret_cast<bf16*>(out.data_ptr()), max_seq, max_splits, (float)scale, (float)eps,
           (int)late_trigger);
  };
  switch (chunk) {
    case 32: run(attention_kernel<32>, 32); break;
    case 64: run(attention_kernel<64>, 64); break;
    case 128: run(attention_kernel<128>, 128); break;
    default: TORCH_CHECK(false, "chunk must be 32, 64 or 128");
  }
}

void lm_head_argmax(torch::Tensor w, torch::Tensor x, torch::Tensor norm_w, torch::Tensor logits,
                    torch::Tensor best, double eps, int64_t ctas) {
  constexpr int WARPS = 8;
  const int V = w.size(0);
  launch(lm_head_argmax_kernel<WARPS>, (int)ctas, WARPS * 32,
         reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const bf16*>(x.data_ptr()),
         reinterpret_cast<const bf16*>(norm_w.data_ptr()), reinterpret_cast<bf16*>(logits.data_ptr()),
         reinterpret_cast<unsigned long long*>(best.data_ptr()), V, (float)eps);
}

void embed(torch::Tensor token, torch::Tensor table, torch::Tensor resid) {
  launch(embed_kernel, 1, HIDDEN / 8, (const int*)token.data_ptr<int>(),
         reinterpret_cast<const bf16*>(table.data_ptr()), reinterpret_cast<bf16*>(resid.data_ptr()));
}

void finalize(torch::Tensor best, torch::Tensor token, torch::Tensor out_tokens, torch::Tensor state) {
  launch(finalize_kernel, 1, 1, reinterpret_cast<unsigned long long*>(best.data_ptr()), token.data_ptr<int>(),
         out_tokens.data_ptr<int>(), state.data_ptr<int>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_gemv", &fp8_gemv);
  m.def("attention", &attention);
  m.def("lm_head_argmax", &lm_head_argmax);
  m.def("embed", &embed);
  m.def("finalize", &finalize);
  m.def("set_pdl", &set_pdl);
  m.def("set_timeline", &set_timeline);
}
