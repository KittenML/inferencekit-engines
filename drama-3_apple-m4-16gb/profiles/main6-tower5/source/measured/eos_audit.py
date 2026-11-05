# SPDX-License-Identifier: Apache-2.0
# Frame loop adapted unchanged from MLX-Audio MIT, fish_drama3/drama3.py,
# commit 76f35931cfe7db8f1f6b604d7b205ffe9d74ff8d. Full MIT notice in licenses/.
"""Observe the upstream EOS scalar already read by the frame loop; no extra GPU evaluation."""
from types import MethodType

def install(model,observations,mx=None,emission_sink=None,segment_sink=None):
    if mx is None:import mlx.core as mx
    def frames(self,frames,system_rows,voice,max_frames,temperature,top_k,top_p):
        caches=self._start_caches(frames,system_rows,voice)
        hidden=self._prefill(frames,caches,system_rows,frames.shape[0])
        args=(caches,temperature,top_k,top_p)
        pending=self._step(hidden,frames[None,-1:],frames[-16:,0],*args)
        mx.async_eval(pending);held=None;emitted=0;steps=0;termination='ABORTED'
        try:
            for n in range(max_frames):
                frame,hidden,history,main=pending;steps=n+1
                if n+1<max_frames:
                    pending=self._step(hidden,frame,history,*args);mx.async_eval(pending)
                token=main.item()
                if token==self.eos_id:
                    termination='EOS';return
                if held is not None:
                    emitted+=1
                    if emission_sink is not None:emission_sink(len(observations),held)
                    yield held
                held=frame[0,0,1:]
                if token!=self.semantic_begin:
                    emitted+=1
                    if emission_sink is not None:emission_sink(len(observations),held)
                    yield held;held=None
            if held is not None:
                emitted+=1
                if emission_sink is not None:emission_sink(len(observations),held)
                yield held
            termination='FRAME_CAP'
        finally:
            observations.append({'segment_index':len(observations),'termination':termination,'eos_observed':termination=='EOS','frame_cap_hit':termination=='FRAME_CAP','generated_steps':steps,'emitted_audio_frames':emitted,'frame_budget':max_frames,'observation':'actual upstream main.item EOS test; unchanged lookahead/held-frame rules'})
            if segment_sink is not None:segment_sink(len(observations)-1)
    model._frames=MethodType(frames,model)
