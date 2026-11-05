# SPDX-License-Identifier: Apache-2.0
"""Prepared pure residency refactor; NOT deployed or qualified by V13 results."""
import gc,hashlib,inspect,json,re
from pathlib import Path
from types import MethodType
CODEC_SOURCE_SHA256='e0045f6df7fb8945f348f31d05071785e0930ed4a44af417364ec7778f42e260'
REFERENCE_MANIFEST_SHA256='5884294cea15c114ad21851412f67c4c128cf60c10796c0c5a91fdafe5a632f7'
def encoder_only(name):
 return name.startswith(('encoder.','quantizer.pre_module.','quantizer.downsample.')) or re.match(r'quantizer\.(semantic_quantizer|quantizer)\.quantizers\.\d+\.in_proj\.',name) is not None

def release(model,reference_cache,mx,*,enabled,audit_payloads=False):
 """Called only after full stock load, fixed hook installation and FP16 refold."""
 from mlx.utils import tree_flatten
 codec=model.codec
 if hashlib.sha256(Path(inspect.getfile(type(codec))).read_bytes()).hexdigest()!=CODEC_SOURCE_SHA256:raise ValueError('Codec source differs from reviewed exact pinned call graph')
 if enabled:
  hook=getattr(model,'_encode_audio',None)
  if reference_cache is None or reference_cache.metadata.get('manifest_sha256')!=REFERENCE_MANIFEST_SHA256 or not isinstance(hook,MethodType) or hook.__self__ is not model or inspect.getclosurevars(hook.__func__).nonlocals.get('registered') is not reference_cache:raise ValueError('Encoder release requires the actual exact-reference cache hook')
 def snapshot():
  items=list(tree_flatten(codec.parameters()))
  for name,module in codec.named_modules():
   value=module.get('_folded')
   if value is not None:items.append((name+'._folded',value))
  metadata={};digest=hashlib.sha256()
  for name,value in items:
   metadata[name]={'object_id':id(value),'shape':list(value.shape),'dtype':str(value.dtype),'bytes':value.nbytes}
   if audit_payloads and not encoder_only(name):
    import numpy as np
    mx.eval(value);array=np.ascontiguousarray(np.asarray(value));digest.update(json.dumps([name,list(value.shape),str(value.dtype)],separators=(',',':')).encode());digest.update(memoryview(array).cast('B'))
  return metadata,digest.hexdigest() if audit_payloads else None
 mx.synchronize();before_active=mx.get_active_memory();before,before_digest=snapshot()
 if enabled:
  codec.pop('encoder');codec.quantizer.pop('pre_module');codec.quantizer.pop('downsample')
  for collection in (codec.quantizer.semantic_quantizer,codec.quantizer.quantizer):
   for quantizer in collection.quantizers:quantizer.pop('in_proj')
 after,after_digest=snapshot();expected={name:value for name,value in before.items() if not(enabled and encoder_only(name))}
 if after!=expected or audit_payloads and before_digest!=after_digest:raise RuntimeError('Retained codec decoder/folding tensors changed')
 removed={name:value for name,value in before.items() if name not in after}
 # Metadata contains integer object IDs, not strong tensor references. Releasing
 # Python objects here cannot keep the removed encoder alive through the audit.
 gc.collect();mx.clear_cache();mx.synchronize()
 return {'policy':'exact-stock-refold-encoder-residency-v1','enabled':enabled,'retained_tensor_objects_shapes_dtypes_unchanged':True,'retained_decoder_folded_payload_sha256_before':before_digest,'retained_decoder_folded_payload_sha256_after':after_digest,'payload_audit_enabled':audit_payloads,'removed_tensor_view_count':len(removed),'removed_logical_tensor_view_bytes':sum(v['bytes'] for v in removed.values()),'tensor_view_accounting':'May include aliases; logical view bytes are not summed physical residency','mlx_active_bytes_before':before_active,'mlx_active_bytes_after':mx.get_active_memory(),'encoder_or_reference_fallback':False,'removed_paths':sorted(removed),'source_sha256':CODEC_SOURCE_SHA256}
