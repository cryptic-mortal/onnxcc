import onnx
from onnx import helper, TensorProto
import numpy as np
import os

np.random.seed(42)
def build_mlp():
    ow1 = np.random.rand(4,8).astype(np.float32)
    ob1 = np.random.rand(8).astype(np.float32)

    ow2 = np.random.rand(8,2).astype(np.float32)
    ob2 = np.random.rand(2).astype(np.float32)

    w1 = helper.make_tensor("w1", TensorProto.FLOAT, [4,8], ow1.flatten().tolist())
    b1 = helper.make_tensor("b1", TensorProto.FLOAT, [8], ob1.flatten().tolist())
    w2 = helper.make_tensor("w2", TensorProto.FLOAT, [8,2], ow2.flatten().tolist())
    b2 = helper.make_tensor("b2", TensorProto.FLOAT,[2], ob2.flatten().tolist())

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1,4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1,2])

    nodes = [
    helper.make_node("MatMul", inputs=["input", "w1"], outputs=["hidden_mm"], name="matmul_1"),
    helper.make_node("Add", inputs=["hidden_mm", "b1"], outputs=["hidden_add"], name="add_1"),
    helper.make_node("Relu", inputs=["hidden_add"], outputs=["hidden_relu"], name="relu_1"),

    helper.make_node("MatMul", inputs=["hidden_relu", "w2"], outputs=["output_mm"], name="matmul_2"),
    helper.make_node("Add", inputs=["output_mm", "b2"], outputs=["output_add"], name="add_2"),
    helper.make_node("Relu", inputs=["output_add"], outputs=["output"], name="relu_2"),
    ]

    graph_def = helper.make_graph(
        nodes=nodes,
        name="mlp_4_8_2",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[w1, b1, w2, b2]
    )
    # Node count: 6 ( matmul, add, relu, matmul, add, relu)
    # initializer count 4 (w1,b1,w2,b2)

    model_def = helper.make_model(
    graph_def,
    producer_name="testmod.py",
    opset_imports=[helper.make_opsetid("", 13)],
    )
    model_def.ir_version = 8
    
    base_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
    path = os.path.join(base_dir, "mlp.onnx")
    return (model_def, path)

def make_input():
    rng = np.random.default_rng(43)
    return rng.standard_normal((1, 4)).astype(np.float32)


def verify(model):
    onnx.checker.check_model(model)
    ops = [n.op_type for n in model.graph.node]
    print("op types:", ops)
    assert set(ops) <= {"MatMul", "Add", "Relu"}, "unexpected op (Gemm?)"
    assert len(model.graph.node) == 6
    assert len(model.graph.initializer) == 4
    assert model.opset_import[0].version == 13


def main():
    model, model_path = build_mlp()
    verify(model)
    onnx.save(model, model_path)

    bin_path = os.path.join(os.path.dirname(model_path), "input.bin")
    make_input().tofile(bin_path)
    assert os.path.getsize(bin_path) == 16
    print("wrote", os.path.basename(model_path), "and", os.path.basename(bin_path))


if __name__ == "__main__":
    main()