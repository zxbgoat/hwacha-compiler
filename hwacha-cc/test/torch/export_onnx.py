#!/usr/bin/env python3
"""ONNX -> linalg-on-tensors MLIR through torch-mlir's ONNX importer, plus an onnxruntime reference point.
   usage: export_onnx.py model.onnx out.mlir out_check.bin [seed]
   The check file: int32 n_inputs, then per input int32 n_elements + floats (a random normal input, or
   the model's declared shape), then the first output's floats. torch_main.c reads the same layout."""
import sys, struct, subprocess, numpy as np, onnx, onnxruntime as ort
path, out, check = sys.argv[1:4]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
m = onnx.load(path)
try:
    v = max((op.version for op in m.opset_import if op.domain in ('', 'ai.onnx')), default=0)
    if v and v < 13: m = onnx.version_converter.convert_version(m, 13); onnx.save(m, path + '.v13'); path = path + '.v13'
except Exception as e: print('opset upgrade skipped:', e)
rng = np.random.default_rng(seed)
sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
inputs = {}
for i in sess.get_inputs():
    shape = [d if isinstance(d, int) else 1 for d in i.shape]
    inputs[i.name] = rng.standard_normal(shape).astype(np.float32)
ref = sess.run(None, inputs)[0]
with open(check, 'wb') as f:
    f.write(struct.pack('i', len(inputs)))
    for v in inputs.values(): f.write(struct.pack('i', v.size)); f.write(v.tobytes())
    f.write(struct.pack('i', ref.size)); f.write(np.ascontiguousarray(ref).astype(np.float32).tobytes())
# ONNX -> torch dialect (onnx.* ops) -> torch backend -> linalg on tensors
tmp = out + '.torch.mlir'
subprocess.check_call(['torch-mlir-import-onnx', path, '-o', tmp])
subprocess.check_call(['torch-mlir-opt', tmp, '--torch-onnx-to-torch-backend-pipeline', '--torch-backend-to-linalg-on-tensors-backend-pipeline', '-o', out])
print("exported", out, "inputs", {k: list(v.shape) for k, v in inputs.items()}, "output", list(ref.shape))
