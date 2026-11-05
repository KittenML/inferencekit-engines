# SPDX-License-Identifier: Apache-2.0
"""Observe actual text passed into the pinned frame builder without recording its plaintext."""
import hashlib
from types import MethodType

def install_input_audit(model,observations,prompt_sink=None,history_sink=None):
    original=model.build_frames
    def build(this,references,history,target):
        result=original(references,history,target)
        index=len(observations)
        observations.append({'segment_index':index,'text_utf8_sha256':hashlib.sha256(target.encode()).hexdigest(),'text_utf8_bytes':len(target.encode()),'prefill_rows_total':int(result[0].shape[0]),'system_rows':int(result[1]),'reference_count':len(references),'history_count':len(history),'history_text_utf8_sha256':[hashlib.sha256(turn[0].encode()).hexdigest() for turn in history]})
        if history_sink is not None:history_sink(index,history,target)
        if prompt_sink is not None:prompt_sink(index,result[0])
        return result
    object.__setattr__(model,'build_frames',MethodType(build,model))
