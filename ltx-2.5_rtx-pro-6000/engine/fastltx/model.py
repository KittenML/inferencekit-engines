"""Batched, exact re-arrangement of the Diffusers LTX-2 transformer forward for one guided denoising step.

The reference (diffusers LTX2Pipeline at 7ecb1e1) runs, per step, three transformer calls:
  1. [uncond, cond] as a batch of 2,
  2. the spatio-temporal-guidance (STG) pass: cond inputs, block 28's self-attention replaced by its value
     projection (video and audio),
  3. the modality-isolation pass: cond inputs, no audio-to-video / video-to-audio cross-attention in any block.

Here one call computes all four rows with the same modules and the same weights:
  * blocks 0..27 run a batch of 3 rows [uncond, cond, modality]; the cross-modal attention runs only on the
    rows that use it (rows 0..1), the modality row skips it as `isolate_modalities=True` does;
  * at block 28 the cond row is duplicated into the STG row (an exact reuse: up to the first
    perturbed block the STG pass and the cond pass compute the same thing on the same inputs), giving
    [uncond, cond, stg, modality]; block 28's self-attention for the STG row is its value projection
    (LTX2PerturbedAttnProcessor with all_perturbed), every other row attends normally;
  * blocks 28..47 run all four rows; cross-modal attention on rows 0..2.
Rows are independent everywhere (per-row attention, per-token linears and norms), so every row computes what
its reference pass computes; only the batch size of the matrix multiplies differs.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_ltx2 import apply_interleaved_rotary_emb, apply_split_rotary_emb

from torch.nn.attention import SDPBackend, sdpa_kernel

# The SDPA backend the transformer's attention uses: "auto" (PyTorch's choice: flash here), or one forced backend.
ATTN = {"backend": "auto"}
_BACKENDS = {"cudnn": SDPBackend.CUDNN_ATTENTION, "flash": SDPBackend.FLASH_ATTENTION,
             "efficient": SDPBackend.EFFICIENT_ATTENTION}


def _sdpa(q, k, v):
    # q, k, v: [B, S, H, D] -> [B, S, H, D], as diffusers' native backend calls it (permute to [B, H, S, D])
    q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))
    be = ATTN["backend"]
    if be == "auto":
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    else:
        with sdpa_kernel([_BACKENDS[be]]):
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    return o.permute(0, 2, 1, 3)


def _rope(attn, x, emb):
    if attn.rope_type == "interleaved":
        return apply_interleaved_rotary_emb(x, emb)
    return apply_split_rotary_emb(x, emb)


LIN = {"split_bias": False}


def _lin(m, x):
    """nn.Linear. With split_bias the bias is added after a plain GEMM (cuBLAS picks a faster kernel than its
    bias-epilogue one here); the sum is then rounded to bf16 twice instead of once."""
    if LIN["split_bias"] and m.bias is not None:
        return F.linear(x, m.weight) + m.bias
    return m(x)


QK = {"fused": False}


def _qk(attn, norm, lin, x, tab):
    """lin(x) then norm (RMSNorm across heads) then split RoPE with tables tab ([T, H, R] cos, sin) or none."""
    if QK["fused"] and attn.rope_type == "split" and isinstance(norm, torch.nn.RMSNorm) and norm.weight is not None:
        from .qknorm import norm_rope

        if LIN["split_bias"]:
            y, b = F.linear(x, lin.weight), lin.bias
        else:
            y, b = lin(x), None
        cs, sn = (tab[0], tab[1]) if tab is not None else (None, None)
        return norm_rope(y, b, norm.weight, cs, sn, float(norm.eps), attn.heads)
    y = norm(_lin(lin, x))
    if tab is not None:
        y = _rope(attn, y, tab[2])
    return y


def attention(attn, hidden_states, encoder_hidden_states=None, q_rope=None, k_rope=None, value_rows=None):
    """LTX2Attention with LTX2PerturbedAttnProcessor semantics, no attention mask.

    q_rope / k_rope: (cos_table, sin_table, (cos, sin) as diffusers gives them, expanded to the batch) or None.
    value_rows: None, or a list of row indices whose output is the value projection (STG perturbation)."""
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    gate_logits = attn.to_gate_logits(hidden_states) if attn.to_gate_logits is not None else None
    value = _lin(attn.to_v, encoder_hidden_states)
    B = hidden_states.shape[0]
    if value_rows:
        keep = [i for i in range(B) if i not in value_rows]
    else:
        keep = None
    hs_q = hidden_states if keep is None else hidden_states[keep]
    enc_k = encoder_hidden_states if keep is None else encoder_hidden_states[keep]
    v_att = value if keep is None else value[keep]
    if keep is not None and len(keep) == 0:  # every row perturbed: the output is the value projection
        out = value
        if gate_logits is not None:
            out = out.unflatten(2, (attn.heads, -1))
            out = out * (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)
            out = out.flatten(2, 3)
        out = _lin(attn.to_out[0], out)
        return attn.to_out[1](out)
    if k_rope is None:
        k_rope = q_rope
    if keep is not None:
        sub = lambda r: None if r is None else (r[0], r[1], tuple(t[keep] for t in r[2]))  # noqa: E731
        q_rope, k_rope = sub(q_rope), sub(k_rope)
    query = _qk(attn, attn.norm_q, attn.to_q, hs_q, q_rope)
    key = _qk(attn, attn.norm_k, attn.to_k, enc_k, k_rope)
    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    v_att = v_att.unflatten(2, (attn.heads, -1))
    out = _sdpa(query, key, v_att)
    out = out.flatten(2, 3).to(query.dtype)
    if keep is not None:
        full = value.clone()
        full[keep] = out
        out = full
    if gate_logits is not None:
        out = out.unflatten(2, (attn.heads, -1))
        gates = 2.0 * torch.sigmoid(gate_logits)
        out = out * gates.unsqueeze(-1)
        out = out.flatten(2, 3)
    out = _lin(attn.to_out[0], out)
    out = attn.to_out[1](out)
    return out


def _mod(table, temb):
    # LTX2VideoTransformerBlock.get_mod_params with batch 1 (temb is the same for every row): [1, 1, D] each
    n = table.shape[0]
    ada = table[None, None].to(temb.device) + temb.reshape(1, temb.shape[1], n, -1)
    return ada.unbind(dim=2)


FF = {"gelu_epilogue": False}


def _ff(ff, x):
    """diffusers FeedForward (GELU(tanh) projection, then Linear). With gelu_epilogue the GELU runs in the
    first GEMM's cuBLASLt epilogue on the fp32 accumulator (one bf16 rounding instead of two)."""
    if not FF["gelu_epilogue"]:
        return ff(x)
    proj = ff.net[0].proj
    bias = proj.bias if proj.bias is not None else _zero_bias(proj.out_features, x)
    sh = x.shape
    h = torch._addmm_activation(bias, x.reshape(-1, sh[-1]), proj.weight.t(), use_gelu=True)
    h = h.reshape(*sh[:-1], -1)
    for m in list(ff.net)[1:]:
        h = m(h)
    return h


_ZB = {}


def _zero_bias(n, x):
    k = (n, x.dtype, x.device)
    if k not in _ZB:
        _ZB[k] = torch.zeros(n, dtype=x.dtype, device=x.device)
    return _ZB[k]


def block_forward(blk, hs, ahs, enc, aenc, c, n_cm, stg_rows):
    """One LTX2VideoTransformerBlock.forward for a batch whose first n_cm rows use the cross-modal attention
    and whose rows in stg_rows are STG-perturbed (self-attention = value projection)."""
    vp = _mod(blk.scale_shift_table, c["temb"])
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = vp[:6]
    if blk.video_cross_attn_adaln:
        shift_text_q, scale_text_q, gate_text_q = vp[6:9]

    nh = blk.norm1(hs)
    nh = nh * (1 + scale_msa) + shift_msa
    a = attention(blk.attn1, nh, None, c["v_rope"], None, stg_rows)
    hs = hs + a * gate_msa

    ap = _mod(blk.audio_scale_shift_table, c["temb_a"])
    a_shift_msa, a_scale_msa, a_gate_msa, a_shift_mlp, a_scale_mlp, a_gate_mlp = ap[:6]
    if blk.audio_cross_attn_adaln:
        a_shift_text_q, a_scale_text_q, a_gate_text_q = ap[6:9]

    nah = blk.audio_norm1(ahs)
    nah = nah * (1 + a_scale_msa) + a_shift_msa
    a = attention(blk.audio_attn1, nah, None, c["a_rope"], None, stg_rows)
    ahs = ahs + a * a_gate_msa

    if blk.cross_attn_adaln:
        if c["temb_p"] is not None:
            shift_text_kv, scale_text_kv = _mod(blk.prompt_scale_shift_table, c["temb_p"])
        else:
            shift_text_kv, scale_text_kv = blk.prompt_scale_shift_table[None, None].to(
                device=hs.device, dtype=hs.dtype).unbind(dim=2)
        if c["temb_pa"] is not None:
            a_shift_text_kv, a_scale_text_kv = _mod(blk.audio_prompt_scale_shift_table, c["temb_pa"])
        else:
            a_shift_text_kv, a_scale_text_kv = blk.audio_prompt_scale_shift_table[None, None].to(
                device=ahs.device, dtype=ahs.dtype).unbind(dim=2)

    nh = blk.norm2(hs)
    if blk.video_cross_attn_adaln:
        nh = nh * (1 + scale_text_q) + shift_text_q
    e = enc
    if blk.cross_attn_adaln:
        e = e * (1 + scale_text_kv) + shift_text_kv
    a = attention(blk.attn2, nh, e)
    if blk.video_cross_attn_adaln:
        a = a * gate_text_q
    hs = hs + a

    nah = blk.audio_norm2(ahs)
    if blk.audio_cross_attn_adaln:
        nah = nah * (1 + a_scale_text_q) + a_shift_text_q
    ae = aenc
    if blk.cross_attn_adaln:
        ae = ae * (1 + a_scale_text_kv) + a_shift_text_kv
    a = attention(blk.audio_attn2, nah, ae)
    if blk.audio_cross_attn_adaln:
        a = a * a_gate_text_q
    ahs = ahs + a

    if n_cm > 0:
        hs_c, ahs_c = hs[:n_cm], ahs[:n_cm]
        nh = blk.audio_to_video_norm(hs_c)
        nah = blk.video_to_audio_norm(ahs_c)

        v_ss = blk.video_a2v_cross_attn_scale_shift_table[:4, :]
        v_g = blk.video_a2v_cross_attn_scale_shift_table[4:, :]
        v_a2v_scale, v_a2v_shift, v_v2a_scale, v_v2a_shift = _mod(v_ss, c["ca_ss"])
        a2v_gate = _mod(v_g, c["ca_gate"])[0].squeeze(2)

        a_ss = blk.audio_a2v_cross_attn_scale_shift_table[:4, :]
        a_g = blk.audio_a2v_cross_attn_scale_shift_table[4:, :]
        a_a2v_scale, a_a2v_shift, a_v2a_scale, a_v2a_shift = _mod(a_ss, c["ca_a_ss"])
        v2a_gate = _mod(a_g, c["ca_a_gate"])[0].squeeze(2)

        m_nh = nh * (1 + v_a2v_scale.squeeze(2)) + v_a2v_shift.squeeze(2)
        m_nah = nah * (1 + a_a2v_scale.squeeze(2)) + a_a2v_shift.squeeze(2)
        a2v = attention(blk.audio_to_video_attn, m_nh, m_nah, c["cav_rope"][n_cm], c["caa_rope"][n_cm])
        hs_c = hs_c + a2v_gate * a2v

        m_nh = nh * (1 + v_v2a_scale.squeeze(2)) + v_v2a_shift.squeeze(2)
        m_nah = nah * (1 + a_v2a_scale.squeeze(2)) + a_v2a_shift.squeeze(2)
        v2a = attention(blk.video_to_audio_attn, m_nah, m_nh, c["caa_rope"][n_cm], c["cav_rope"][n_cm])
        ahs_c = ahs_c + v2a_gate * v2a

        if n_cm < hs.shape[0]:
            hs = torch.cat([hs_c, hs[n_cm:]], dim=0)
            ahs = torch.cat([ahs_c, ahs[n_cm:]], dim=0)
        else:
            hs, ahs = hs_c, ahs_c

    nh = blk.norm3(hs) * (1 + scale_mlp) + shift_mlp
    hs = hs + _ff(blk.ff, nh) * gate_mlp
    nah = blk.audio_norm3(ahs) * (1 + a_scale_mlp) + a_shift_mlp
    ahs = ahs + blk.audio_ff(nah) * a_gate_mlp
    return hs, ahs


BLOCK = {"fn": block_forward}
GROUPING = {"mode": "batched"}


def enable_compile():
    """torch.compile the block function (one graph per batch size / STG variant). Inductor is told to round
    to bf16 wherever eager rounds (emulate_precision_casts), so fused elementwise chains keep eager's numerics."""
    import torch._dynamo
    import torch._inductor.config as icfg

    icfg.emulate_precision_casts = True
    if os.environ.get("FASTLTX_CDT", "0") == "1":
        icfg.coordinate_descent_tuning = True
    torch._dynamo.config.cache_size_limit = 64
    BLOCK["fn"] = torch.compile(block_forward, dynamic=False, fullgraph=False)


def _expand_rope(emb, b):
    return tuple(t.expand(b, *t.shape[1:]) for t in emb)


class StepRunner:
    """Per clip: rotary embeddings and coords; per step: one batched forward giving the four guidance passes."""

    def __init__(self, tr, latent_num_frames, latent_height, latent_width, fps, audio_num_frames, device,
                 stg_blocks=(28,)):
        self.tr = tr
        cfg = tr.config
        assert not cfg.use_prompt_embeddings
        assert len(stg_blocks) == 1, "one STG block supported"
        self.stg_block = int(stg_blocks[0])
        self.F, self.H, self.W, self.fps, self.La = latent_num_frames, latent_height, latent_width, fps, audio_num_frames
        vc = tr.rope.prepare_video_coords(1, latent_num_frames, latent_height, latent_width, device, fps=fps)
        ac = tr.audio_rope.prepare_audio_coords(1, audio_num_frames, device)
        self.video_coords, self.audio_coords = vc, ac
        v_rope = tr.rope(vc, device=device)
        a_rope = tr.audio_rope(ac, device=device)
        cav = tr.cross_attn_rope(vc[:, 0:1, :], device=device)
        caa = tr.cross_attn_audio_rope(ac[:, 0:1, :], device=device)
        from .qknorm import tables

        tabs = [tables(e) for e in (v_rope, a_rope, cav, caa)]
        self.ropes = {}
        for b in (1, 2, 3, 4):
            self.ropes[b] = tuple((tb[0], tb[1], _expand_rope(e, b)) for tb, e in zip(tabs, (v_rope, a_rope, cav, caa)))

    def temb(self, timestep, sigma, batch=1):
        """Timestep embeddings for one row (every row shares the step's timestep), computed at batch size
        ``batch`` (the reference's batch for that pass) and returned for row 0."""
        tr = self.tr
        dt = torch.bfloat16
        cfg = tr.config
        if batch != 1:
            full = self._temb(timestep.reshape(batch), sigma.reshape(batch), batch)
            return {k: (v[:1] if v is not None else None) for k, v in full.items()}
        return self._temb(timestep.reshape(1), sigma.reshape(1), 1)

    def _temb(self, timestep, sigma, n):
        tr = self.tr
        dt = torch.bfloat16
        cfg = tr.config
        f = cfg.cross_attn_timestep_scale_multiplier / cfg.timestep_scale_multiplier
        temb, emb_t = tr.time_embed(timestep.flatten(), batch_size=n, hidden_dtype=dt)
        temb = temb.view(n, -1, temb.size(-1))
        emb_t = emb_t.view(n, -1, emb_t.size(-1))
        temb_a, emb_ta = tr.audio_time_embed(timestep.flatten(), batch_size=n, hidden_dtype=dt)
        temb_a = temb_a.view(n, -1, temb_a.size(-1))
        emb_ta = emb_ta.view(n, -1, emb_ta.size(-1))
        if tr.prompt_modulation and cfg.use_prompt_adaln_single:
            tp, _ = tr.prompt_adaln(sigma.flatten(), batch_size=n, hidden_dtype=dt)
            tpa, _ = tr.audio_prompt_adaln(sigma.flatten(), batch_size=n, hidden_dtype=dt)
            tp = tp.view(n, -1, tp.size(-1))
            tpa = tpa.view(n, -1, tpa.size(-1))
        else:
            tp = tpa = None
        # use_cross_timestep=True (the pipeline default): video's cross-attn timestep is the audio sigma, which
        # equals the video one here (audio_sigma defaults to sigma)
        v_ca_t = sigma.flatten()
        ca_ss, _ = tr.av_cross_attn_video_scale_shift(v_ca_t, batch_size=n, hidden_dtype=dt)
        ca_gate, _ = tr.av_cross_attn_video_a2v_gate(v_ca_t * f, batch_size=n, hidden_dtype=dt)
        a_ca_t = sigma.flatten()
        ca_a_ss, _ = tr.av_cross_attn_audio_scale_shift(a_ca_t, batch_size=n, hidden_dtype=dt)
        ca_a_gate, _ = tr.av_cross_attn_audio_v2a_gate(a_ca_t * f, batch_size=n, hidden_dtype=dt)
        return dict(temb=temb, temb_a=temb_a, emb_t=emb_t, emb_ta=emb_ta, temb_p=tp, temb_pa=tpa,
                    ca_ss=ca_ss.view(n, -1, ca_ss.shape[-1]), ca_gate=ca_gate.view(n, -1, ca_gate.shape[-1]),
                    ca_a_ss=ca_a_ss.view(n, -1, ca_a_ss.shape[-1]),
                    ca_a_gate=ca_a_gate.view(n, -1, ca_a_gate.shape[-1]))

    def _block(self, blk, hs, ahs, enc, aenc, c, n_cm, stg_rows):
        B = hs.shape[0]
        v_rope, a_rope, cav, caa = self.ropes[B]
        c = dict(c)
        c["v_rope"], c["a_rope"] = v_rope, a_rope
        if n_cm > 0:
            c["cav_rope"] = {n_cm: self.ropes[n_cm][2]}
            c["caa_rope"] = {n_cm: self.ropes[n_cm][3]}
        return BLOCK["fn"](blk, hs, ahs, enc, aenc, c, n_cm, stg_rows)

    def _out(self, hs, ahs, c):
        tr = self.tr
        ssv = tr.scale_shift_table[None, None] + c["emb_t"][:, :, None]
        shift, scale = ssv[:, :, 0], ssv[:, :, 1]
        hs = tr.norm_out(hs)
        hs = hs * (1 + scale) + shift
        out = tr.proj_out(hs)
        assv = tr.audio_scale_shift_table[None, None] + c["emb_ta"][:, :, None]
        a_shift, a_scale = assv[:, :, 0], assv[:, :, 1]
        ahs = tr.audio_norm_out(ahs)
        ahs = ahs * (1 + a_scale) + a_shift
        return out, tr.audio_proj_out(ahs)

    def forward_ref(self, x_v, x_a, enc_neg, enc_pos, aenc_neg, aenc_pos, timestep, sigma):
        """The reference's grouping of rows: [uncond, cond] as a batch of 2, the modality pass and the STG pass
        as batches of 1 (the STG pass starting at block 28 from the cond row's activations), so every matrix
        multiply has the shape the reference gives it. Timestep embeddings are computed per group at the
        group's batch size, as the reference computes them."""
        tr = self.tr
        c2 = self.temb(timestep.expand(2), sigma.expand(2), batch=2)
        c1 = self.temb(timestep, sigma, batch=1)
        hA = tr.proj_in(torch.cat([x_v, x_v], dim=0))
        aA = tr.audio_proj_in(torch.cat([x_a, x_a], dim=0))
        hM = tr.proj_in(x_v)
        aM = tr.audio_proj_in(x_a)
        eA = torch.cat([enc_neg, enc_pos], dim=0)
        aeA = torch.cat([aenc_neg, aenc_pos], dim=0)
        hS = aS = None
        for idx, blk in enumerate(tr.transformer_blocks):
            if idx == self.stg_block:
                hS, aS = hA[1:2].clone(), aA[1:2].clone()
            hA_n, aA_n = self._block(blk, hA, aA, eA, aeA, c2, 2, None)
            if hS is not None:
                hS, aS = self._block(blk, hS, aS, enc_pos, aenc_pos, c1, 1, [0] if idx == self.stg_block else None)
            hM, aM = self._block(blk, hM, aM, enc_pos, aenc_pos, c1, 0, None)
            hA, aA = hA_n, aA_n
        oA, oaA = self._out(hA, aA, c2)
        oS, oaS = self._out(hS, aS, c1)
        oM, oaM = self._out(hM, aM, c1)
        return torch.cat([oA, oS, oM], dim=0), torch.cat([oaA, oaS, oaM], dim=0)

    def forward(self, x_v, x_a, enc_neg, enc_pos, aenc_neg, aenc_pos, timestep, sigma):
        if GROUPING["mode"] == "reference":
            return self.forward_ref(x_v, x_a, enc_neg, enc_pos, aenc_neg, aenc_pos, timestep, sigma)
        return self.forward_batched(x_v, x_a, enc_neg, enc_pos, aenc_neg, aenc_pos, timestep, sigma)

    def forward_batched(self, x_v, x_a, enc_neg, enc_pos, aenc_neg, aenc_pos, timestep, sigma):
        """x_v [1, Sv, Cin] and x_a [1, Sa, Ca] (bf16). Returns (video, audio) outputs, each [4, S, C] with rows
        [uncond, cond, stg, modality]."""
        tr = self.tr
        c = self.temb(timestep, sigma)
        hs = tr.proj_in(x_v)
        ahs = tr.audio_proj_in(x_a)
        hs = hs.expand(3, -1, -1).contiguous()
        ahs = ahs.expand(3, -1, -1).contiguous()
        enc = torch.cat([enc_neg, enc_pos, enc_pos], dim=0)
        aenc = torch.cat([aenc_neg, aenc_pos, aenc_pos], dim=0)
        n_cm = 2
        for idx, blk in enumerate(tr.transformer_blocks):
            stg_rows = None
            if idx == self.stg_block:
                # branch: [uncond, cond, mod] -> [uncond, cond, stg(=cond so far), mod]
                hs = torch.cat([hs[:2], hs[1:2], hs[2:]], dim=0)
                ahs = torch.cat([ahs[:2], ahs[1:2], ahs[2:]], dim=0)
                enc = torch.cat([enc[:2], enc[1:2], enc[2:]], dim=0)
                aenc = torch.cat([aenc[:2], aenc[1:2], aenc[2:]], dim=0)
                n_cm = 3
                stg_rows = [2]
            B = hs.shape[0]
            v_rope, a_rope, cav, caa = self.ropes[B]
            c["v_rope"], c["a_rope"] = v_rope, a_rope
            c["cav_rope"] = {n_cm: self.ropes[n_cm][2]}
            c["caa_rope"] = {n_cm: self.ropes[n_cm][3]}
            hs, ahs = BLOCK["fn"](blk, hs, ahs, enc, aenc, c, n_cm, stg_rows)

        ssv = tr.scale_shift_table[None, None] + c["emb_t"][:, :, None]
        shift, scale = ssv[:, :, 0], ssv[:, :, 1]
        hs = tr.norm_out(hs)
        hs = hs * (1 + scale) + shift
        out = tr.proj_out(hs)
        assv = tr.audio_scale_shift_table[None, None] + c["emb_ta"][:, :, None]
        a_shift, a_scale = assv[:, :, 0], assv[:, :, 1]
        ahs = tr.audio_norm_out(ahs)
        ahs = ahs * (1 + a_scale) + a_shift
        aout = tr.audio_proj_out(ahs)
        return out, aout
