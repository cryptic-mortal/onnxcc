#generate_golden.py - produces golden output for the ONNX runtime
#Requirements- onnxruntime==1.30.0, onnx, numpy


import numpy as np
import onnx
from onnx import helper, TensorProto
import argparse, json, re
from pathlib import Path
import onnxruntime as ort


PINNED_ORT="1.30.0"
#preserve same version to maintain golden files being byte identical across machines
if ort.__version__ !=PINNED_ORT:
    raise SystemExit(f"Expected onnxruntime {PINNED_ORT},found {ort.__version__}")

p=argparse.ArgumentParser()

#model and input file are compulsory
p.add_argument("model")
p.add_argument("input")
p.add_argument("--out-dir",default=".")

#without the intermediate only the final output stored
p.add_argument("--intermediates",action="store_true")

args=p.parse_args()
out_dir=Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

#creates settings object for the inference session
so=ort.SessionOptions()

#prevents default optimizations of ORT which may change matmul+add to a Gemm node
so.graph_optimization_level= ort.GraphOptimizationLevel.ORT_DISABLE_ALL

#control parallelism and ensure every run does arithmetic in same order
so.intra_op_num_threads=1
so.inter_op_num_threads=1
so.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL

model=onnx.load(args.model)
if args.intermediates:
    model=onnx.shape_inference.infer_shapes(model)
    g=model.graph
    known={vi.name: vi for vi in g.value_info}
    existing={o.name for o in g.output}
    for node in g.node:
        for name in node.output:
            if name and name not in existing:
                vi=known.get(name) or helper.make_tensor_value_info(name,TensorProto.FLOAT,None)#fallback runs when shape inference produces no entry for that tensor
                g.output.append(vi) 
                existing.add(name)

sess= ort.InferenceSession(model.SerializeToString(),so,providers=["CPUExecutionProvider"])

#translation of data types between ORT and numpy
DTYPES = {"tensor(float)": np.float32, "tensor(double)": np.float64,
          "tensor(int64)": np.int64, "tensor(int32)": np.int32}

inp = sess.get_inputs()[0]
dtype = DTYPES[inp.type]#raises KeyError if model used a type not in dictionary
shape = [d if isinstance(d, int) else -1 for d in inp.shape]

x = np.fromfile(args.input, dtype=dtype).reshape(shape)#reshape catches mismatched input file

names = [o.name for o in sess.get_outputs()]
results = sess.run(names, {inp.name: x})

def safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)

meta = []
for name, arr in zip(names, results):
    #guarantees tofile writes bytes in the order that C++ would read them
    arr = np.ascontiguousarray(arr)
    fname = f"expected_{safe(name)}.bin"
    arr.tofile(out_dir / fname)
    meta.append({"name": name, "file": fname,
                 "shape": list(arr.shape), "dtype": str(arr.dtype)})

#the json sidecar
with open(out_dir / "expected.json", "w") as f:
    json.dump({"outputs": meta}, f, indent=2, sort_keys=True)
    f.write("\n")


