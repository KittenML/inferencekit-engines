"""One text-to-video clip with LTX-2.5 (full BF16 dev model), computed as the Diffusers LTX2Pipeline computes it.

Loading, prompt encoding, latent preparation, the sigma schedule, the guidance arithmetic, the Euler step and
the decoders are the Diffusers pipeline's own methods and modules; only the four transformer passes of a step
run as one batched call (model.StepRunner)."""

from __future__ import annotations

import copy
import hashlib
import os
import time

import numpy as np
import torch

from . import model as fm

DIFFUSERS_REPO = "Lightricks/LTX-2.5-Diffusers"
DIFFUSERS_REV = "da0b690960026eab794b8ceaf841233cf7064478"

# Diffusers LTX2Pipeline defaults at 7ecb1e1 (the workload)
STG_SCALE = 1.0
MODALITY_SCALE = 3.0
GUIDANCE_RESCALE = 0.7
AUDIO_GUIDANCE_SCALE = 7.0
AUDIO_STG_SCALE = 1.0
AUDIO_MODALITY_SCALE = 3.0
AUDIO_GUIDANCE_RESCALE = 0.7
STG_BLOCKS = [28]
MAX_SEQ = 1024


def combine(p, latents, audio_latents, out_v, out_a, i, sched, a_sched, gs, ags=AUDIO_GUIDANCE_SCALE):
    """The pipeline's guidance arithmetic (float32, x0 space, rescale), from the four passes' outputs with rows
    [uncond, cond, stg, modality]; returns the guided velocities the Euler step takes."""
    from diffusers.pipelines.ltx2.pipeline_ltx2 import rescale_noise_cfg

    out_v = out_v.float()
    out_a = out_a.float()
    vu, vc, vs, vm = (out_v[j:j + 1] for j in range(4))
    au, ac, as_, am = (out_a[j:j + 1] for j in range(4))

    vc = p.convert_velocity_to_x0(latents, vc, i, sched)
    vu = p.convert_velocity_to_x0(latents, vu, i, sched)
    v_cfg = (gs - 1) * (vc - vu)
    ac = p.convert_velocity_to_x0(audio_latents, ac, i, a_sched)
    au = p.convert_velocity_to_x0(audio_latents, au, i, a_sched)
    a_cfg = (ags - 1) * (ac - au)

    vs = p.convert_velocity_to_x0(latents, vs, i, sched)
    as_ = p.convert_velocity_to_x0(audio_latents, as_, i, a_sched)
    v_stg = STG_SCALE * (vc - vs)
    a_stg = AUDIO_STG_SCALE * (ac - as_)

    vm = p.convert_velocity_to_x0(latents, vm, i, sched)
    am = p.convert_velocity_to_x0(audio_latents, am, i, a_sched)
    v_mod = (MODALITY_SCALE - 1) * (vc - vm)
    a_mod = (AUDIO_MODALITY_SCALE - 1) * (ac - am)

    vg = vc + v_cfg + v_stg + v_mod
    ag = ac + a_cfg + a_stg + a_mod
    vg = rescale_noise_cfg(vg, vc, guidance_rescale=GUIDANCE_RESCALE)
    ag = rescale_noise_cfg(ag, ac, guidance_rescale=AUDIO_GUIDANCE_RESCALE)
    v_vel = p.convert_x0_to_velocity(latents, vg, i, sched)
    a_vel = p.convert_x0_to_velocity(audio_latents, ag, i, a_sched)
    return v_vel, a_vel


class Stages:
    def __init__(self):
        self.ms = {}
        self.t = time.perf_counter()

    def mark(self, name, sync=True):
        if sync:
            torch.cuda.synchronize()
        now = time.perf_counter()
        self.ms[name] = self.ms.get(name, 0.0) + (now - self.t) * 1000.0
        self.t = now


PATH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".venv", "ltx25_diffusers_snapshot")


def snapshot():
    """The pinned Diffusers snapshot folder in the Hugging Face cache (offline). INSTALL.sh records the folder it
    resolved in .venv/ltx25_diffusers_snapshot, used when the server's environment does not point at the cache."""
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(DIFFUSERS_REPO, revision=DIFFUSERS_REV, local_files_only=True)
    except Exception:  # noqa: BLE001 - e.g. HF_HOME not set in the server's environment
        p = os.environ.get("FASTLTX_SNAPSHOT") or (open(PATH_FILE).read().strip() if os.path.isfile(PATH_FILE) else "")
        if p and os.path.isdir(p) and os.path.basename(os.path.normpath(p)) == DIFFUSERS_REV:
            return p
        raise


class Engine:
    def __init__(self, device="cuda"):
        import diffusers
        from diffusers import LTX2Pipeline, LTX2VideoTransformer3DModel
        from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT

        self.device = torch.device(device)
        root = snapshot()
        dt = torch.bfloat16
        tr = LTX2VideoTransformer3DModel.from_pretrained(root, subfolder="transformer_full", torch_dtype=dt)
        skipped = [c for c in ("prompt_enhancer", "processor", "duration_head")
                   if c in (getattr(LTX2Pipeline, "_optional_components", None) or ())]
        pipe = LTX2Pipeline.from_pretrained(root, transformer=tr, torch_dtype=dt, **{c: None for c in skipped})
        pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config, use_dynamic_shifting=True,
                                                          shift_terminal=0.1)
        pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.negative_prompt = DEFAULT_NEGATIVE_PROMPT
        self.diffusers_version = diffusers.__version__
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self._neg = None
        fm.ATTN["backend"] = os.environ.get("FASTLTX_ATTN", "auto")
        self.attention_name = "sdpa:" + fm.ATTN["backend"]
        if os.environ.get("FASTLTX_GELU_EPI", "0") == "1":
            fm.FF["gelu_epilogue"] = True
            self.attention_name += "+gelu_epilogue"
        if os.environ.get("FASTLTX_SPLIT_BIAS", "0") == "1":
            fm.LIN["split_bias"] = True
            self.attention_name += "+split_bias"
        if os.environ.get("FASTLTX_QK_FUSED", "0") == "1":
            fm.QK["fused"] = True
            self.attention_name += "+fused_qk_norm_rope"
        if os.environ.get("FASTLTX_GROUPING", "batched") == "reference":
            fm.GROUPING["mode"] = "reference"
            self.attention_name += "+reference_grouping"
        if os.environ.get("FASTLTX_COMPILE", "0") == "1":
            fm.enable_compile()
            self.attention_name += "+compiled"

    # ------------------------------------------------------------------ text
    @torch.inference_mode()
    def _encode(self, text):
        p = self.pipe
        emb, mask = p._get_gemma_prompt_embeds(prompt=text, num_videos_per_prompt=1, max_sequence_length=MAX_SEQ,
                                               device=self.device)
        return emb, mask

    @torch.inference_mode()
    def _connect(self, emb, mask):
        p = self.pipe
        side = getattr(p.tokenizer, "padding_side", "left") if getattr(p, "tokenizer", None) is not None else "left"
        v, a, m = p.connectors(emb, mask, padding_side=side)
        return v, a, m

    @torch.inference_mode()
    def negative(self):
        """The fixed negative prompt's encoding (Gemma + connectors), computed once."""
        if self._neg is None:
            emb, mask = self._encode(self.negative_prompt)
            self._neg = self._connect(emb, mask)
        return self._neg

    # ------------------------------------------------------------------ shapes
    def shapes(self, width, height, num_frames, fps):
        p = self.pipe
        s, t = p.vae_spatial_compression_ratio, p.vae_temporal_compression_ratio
        video = (1, p.transformer.config.in_channels, (num_frames - 1) // t + 1, height // s, width // s)
        per_s = p.audio_sampling_rate / p.audio_hop_length / float(p.audio_vae_temporal_compression_ratio)
        length = round(num_frames / fps * per_s)
        audio = (1, p.audio_vae.config.latent_channels, length, p.audio_vae.config.mel_bins //
                 p.audio_vae_mel_compression_ratio)
        return {"video": video, "audio": audio}

    # ------------------------------------------------------------------ one clip
    @torch.inference_mode()
    def generate(self, prompt, width, height, num_frames, fps, steps, guidance_scale, noise_seed, recorder=None,
                 on_denoised=None):
        p = self.pipe
        dev = self.device
        st = Stages()
        shapes = self.shapes(width, height, num_frames, fps)
        # initial noise: CPU generator, float32, video then audio (the pipeline's own first two draws)
        gen = torch.Generator(device="cpu").manual_seed(int(noise_seed))
        noise_v = torch.randn(shapes["video"], generator=gen, dtype=torch.float32)
        noise_a = torch.randn(shapes["audio"], generator=gen, dtype=torch.float32)

        # text: the prompt's Gemma encoding and connectors; the negative prompt's encoding computed once
        neg_v, neg_a, neg_m = self.negative()
        emb, mask = self._encode(prompt)
        pos_v, pos_a, pos_m = self._connect(emb, mask)
        if not (bool(neg_m.all()) and bool(pos_m.all())):
            raise RuntimeError("connector attention mask is not all ones; masked cross-attention not implemented")
        st.mark("text_encode")

        _, C, Fl, Hl, Wl = shapes["video"]
        _, Ca, La, Ma = shapes["audio"]
        latents = p._pack_latents(noise_v.to(dev), p.transformer_spatial_patch_size,
                                  p.transformer_temporal_patch_size)
        audio_latents = p._pack_audio_latents(noise_a.to(dev))

        sched = copy.deepcopy(self._sched_proto())
        a_sched = copy.deepcopy(sched)
        from diffusers.pipelines.ltx2.pipeline_ltx2 import calculate_shift, retrieve_timesteps
        sigmas = np.linspace(1.0, 1 / steps, steps)
        mu = calculate_shift(latents.shape[1], sched.config.get("base_image_seq_len", 1024),
                             sched.config.get("max_image_seq_len", 4096), sched.config.get("base_shift", 0.95),
                             sched.config.get("max_shift", 2.05))
        retrieve_timesteps(a_sched, steps, dev, None, sigmas=sigmas, mu=mu)
        timesteps, _ = retrieve_timesteps(sched, steps, dev, None, sigmas=sigmas, mu=mu)
        sched.set_begin_index(0)
        a_sched.set_begin_index(0)

        runner = fm.StepRunner(p.transformer, Fl, Hl, Wl, float(fps), La, dev, STG_BLOCKS)
        st.mark("prepare_latents")

        sig_cpu = [float(v) for v in sched.sigmas.cpu()]
        gs, ags = float(guidance_scale), AUDIO_GUIDANCE_SCALE
        steps_run = 0
        for i, t in enumerate(timesteps):
            x_in = latents.to(torch.bfloat16)
            xa_in = audio_latents.to(torch.bfloat16)
            ts = t.expand(1)
            out_v, out_a = runner.forward(x_in, xa_in, neg_v, pos_v, neg_a, pos_a, ts, ts)
            v_vel, a_vel = combine(p, latents, audio_latents, out_v, out_a, i, sched, a_sched, gs)

            sig, sig_next = sig_cpu[i], sig_cpu[i + 1]
            new_latents = sched.step(v_vel, t, latents, return_dict=False)[0]
            new_audio = a_sched.step(a_vel, t, audio_latents, return_dict=False)[0]
            if recorder is not None:
                recorder.step("video", sig, sig_next,
                              p._unpack_latents(latents, Fl, Hl, Wl, 1, 1),
                              p._unpack_latents(v_vel, Fl, Hl, Wl, 1, 1),
                              p._unpack_latents(new_latents, Fl, Hl, Wl, 1, 1))
                recorder.step("audio", sig, sig_next,
                              p._unpack_audio_latents(audio_latents, La, num_mel_bins=Ma),
                              p._unpack_audio_latents(a_vel, La, num_mel_bins=Ma),
                              p._unpack_audio_latents(new_audio, La, num_mel_bins=Ma))
            latents, audio_latents = new_latents, new_audio
            steps_run += 1
        st.mark("denoise")

        # decode, as the pipeline does after its loop
        latents = p._unpack_latents(latents, Fl, Hl, Wl, p.transformer_spatial_patch_size,
                                    p.transformer_temporal_patch_size)
        audio_latents = p._denormalize_audio_latents(audio_latents, p.audio_vae.latents_mean,
                                                     p.audio_vae.latents_std)
        audio_latents = p._unpack_audio_latents(audio_latents, La, num_mel_bins=Ma)
        writer = None
        if recorder is not None:
            recorder.final("video", recorder.tensors[f"step.{steps_run - 1}.video.x_next"])
            recorder.final("audio", recorder.tensors[f"step.{steps_run - 1}.audio.x_next"])
            if on_denoised is not None:
                torch.cuda.synchronize()
                writer = on_denoised(recorder)  # e.g. a thread that copies the record to the CPU and writes it
        latents = latents.to(torch.bfloat16)
        if not p.vae.config.timestep_conditioning:
            timestep = None
        else:
            raise RuntimeError("timestep-conditioned VAE decode not implemented")
        latents = p._denormalize_latents(latents, p.vae.latents_mean, p.vae.latents_std, p.vae.config.scaling_factor)
        latents = latents.to(p.vae.dtype)
        video = p.vae.decode(latents, timestep, return_dict=False)[0]
        st.mark("vae_decode")
        video = p.video_processor.postprocess_video(video, output_type="pt")  # [B, F, C, H, W] in [0, 1]
        frames = video[0].permute(0, 2, 3, 1)  # [F, H, W, C]
        frames_u8 = (frames.float() * 255).round().to(torch.uint8).cpu()
        st.mark("postprocess")
        audio_latents = audio_latents.to(p.audio_vae.dtype)
        mel = p.audio_vae.decode(audio_latents, return_dict=False)[0]
        wave = p.vocoder(mel)
        wave = wave[0].float().cpu()
        rate = int(p.vocoder.config.output_sampling_rate)
        st.mark("audio_decode")
        return dict(frames=frames_u8, wave=wave, rate=rate, steps_run=steps_run, stages=st, writer=writer,
                    shape=(int(frames_u8.shape[0]), int(frames_u8.shape[1]), int(frames_u8.shape[2])))

    def _sched_proto(self):
        return self.pipe.scheduler

    def warmup(self):
        """One small clip (not kept) so the first request does not pay for kernel loading."""
        w = os.environ.get("FASTLTX_WARMUP", "768x512x121")
        if w:
            W, H, N = (int(v) for v in w.split("x"))
            self.generate("warm-up", W, H, N, 24.0, 2, 3.0, 0, recorder=None)
