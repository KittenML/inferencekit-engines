# SPDX-License-Identifier: Apache-2.0
import unittest
import mlx.core as mx
from mse_quant import quantize_mse_clip_v1

class QuantizationTest(unittest.TestCase):
    def test_stored_reconstruction_never_increases_error(self):
        for dtype in (mx.float32,mx.float16,mx.bfloat16):
            for group in (32,64,128):
                mx.random.seed(123)
                source=mx.random.normal((2,3,256)).astype(dtype)
                q,s,b,stats=quantize_mse_clip_v1(source,group,row_batch=2)
                self.assertEqual(q.shape,(2,3,16));self.assertEqual(s.shape,(2,3,256//group))
                self.assertEqual(s.dtype,dtype);self.assertEqual(b.dtype,dtype)
                self.assertLessEqual(stats['selected_mse'],stats['baseline_mse']+1e-12)
                decoded=mx.dequantize(q,s.astype(mx.float32),b.astype(mx.float32),group_size=group,bits=2)
                measured=float(mx.mean(mx.square(source.astype(mx.float32)-decoded)).item())
                self.assertAlmostEqual(measured,stats['selected_mse'],places=6)
    def test_outlier_groups_improve_and_constant_is_preserved(self):
        values=mx.concatenate([mx.full((1,127),0.1),mx.array([[2.0]])],axis=-1)
        _,_,_,stats=quantize_mse_clip_v1(values)
        self.assertLessEqual(stats['selected_mse'],stats['baseline_mse'])
        source=mx.full((2,128),0.0,dtype=mx.bfloat16)
        q,s,b,stats=quantize_mse_clip_v1(source)
        self.assertEqual(stats['selected_mse'],0);self.assertEqual(stats['improved_groups'],0)
    def test_rejects_invalid_source_and_group(self):
        for value,group in [(mx.ones((2,128)),48),(mx.ones((2,127)),128),(mx.array([[float('nan')]*128]),128),(mx.ones((2,128),dtype=mx.int32),128)]:
            with self.assertRaises(ValueError):quantize_mse_clip_v1(value,group)

if __name__=='__main__':unittest.main()
