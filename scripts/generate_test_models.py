import onnx
from onnx import helper, TensorProto
import numpy as np
import os

np.random.seed(42)

def create_tensor(name, array):
    return helper.make_tensor(name, TensorProto.FLOAT, list(array.shape), array.flatten().tolist())


# MLP

def build_mlp():
    ow1 = np.random.rand(4,8).astype(np.float32)
    ob1 = np.random.rand(8).astype(np.float32)

    ow2 = np.random.rand(8,2).astype(np.float32)
    ob2 = np.random.rand(2).astype(np.float32)

    w1 = create_tensor("w1", ow1)
    b1 = create_tensor("b1", ob1)
    w2 = create_tensor("w2", ow2)
    b2 = create_tensor("b2", ob2)

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


def verify_mlp(model):
    onnx.checker.check_model(model)

    ops = [n.op_type for n in model.graph.node]

    print("MLP op types:", ops)

    assert set(ops) <= {"MatMul", "Add", "Relu"}, "unexpected op (Gemm?)"
    assert len(model.graph.node) == 6
    assert len(model.graph.initializer) == 4
    assert model.opset_import[0].version == 13


# LeNet

def build_lenet():
    conv1_w_arr = np.random.rand(6,1,5,5).astype(np.float32)
    conv1_b_arr = np.random.rand(6).astype(np.float32)

    conv2_w_arr = np.random.rand(16,6,5,5).astype(np.float32)
    conv2_b_arr = np.random.rand(16).astype(np.float32)

    fc1_w_arr = np.random.rand(256,120).astype(np.float32)
    fc1_b_arr = np.random.rand(120).astype(np.float32)

    fc2_w_arr = np.random.rand(120,10).astype(np.float32)
    fc2_b_arr = np.random.rand(10).astype(np.float32)

    conv1_w = create_tensor("conv1_w", conv1_w_arr)
    conv1_b = create_tensor("conv1_b", conv1_b_arr)

    conv2_w = create_tensor("conv2_w", conv2_w_arr)
    conv2_b = create_tensor("conv2_b", conv2_b_arr)

    fc1_w = create_tensor("fc1_w", fc1_w_arr)
    fc1_b = create_tensor("fc1_b", fc1_b_arr)

    fc2_w = create_tensor("fc2_w", fc2_w_arr)
    fc2_b = create_tensor("fc2_b", fc2_b_arr)

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1,1,28,28])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1,10])

    nodes = [
        # conv 1: [1, 1, 28, 28] -> [1, 6, 24, 24]
        helper.make_node(
            "Conv", inputs=["input", "conv1_w", "conv1_b"], outputs=["conv1_out"], name="conv1", 
            kernel_shape=[5,5], strides=[1,1],
        ),

        helper.make_node("Relu", inputs=["conv1_out"], outputs=["relu1_out"], name="relu1"),

        helper.make_node(
            "MaxPool", inputs=["relu1_out"], outputs=["pool1_out"], name="pool1", 
            kernel_shape=[2,2], strides=[2,2],
        ),

        # conv 2: [1, 6, 12, 12] -> [1, 16, 8, 8]
        helper.make_node(
            "Conv", inputs=["pool1_out", "conv2_w", "conv2_b"], outputs=["conv2_out"], name="conv2", 
            kernel_shape=[5,5], strides=[1,1],
        ),

        helper.make_node("Relu", inputs=["conv2_out"], outputs=["relu2_out"], name="relu2"),

        helper.make_node(
            "MaxPool", inputs=["relu2_out"], outputs=["pool2_out"], name="pool2", 
            kernel_shape=[2,2], strides=[2,2],
        ),

        # [1, 16, 4, 4] -> [1, 256]
        helper.make_node("Flatten", inputs=["pool2_out"], outputs=["flatten_out"], name="flatten", axis=1),

        helper.make_node("MatMul", inputs=["flatten_out", "fc1_w"], outputs=["fc1_mm"], name="matmul_1"),
        helper.make_node("Add", inputs=["fc1_mm", "fc1_b"], outputs=["fc1_add"], name="add_1"),
        helper.make_node("Relu", inputs=["fc1_add"], outputs=["fc1_relu"], name="relu_3"),

        helper.make_node("MatMul", inputs=["fc1_relu", "fc2_w"], outputs=["fc2_mm"], name="matmul_2"),
        helper.make_node("Add", inputs=["fc2_mm", "fc2_b"], outputs=["output"], name="add_2"),
    ]

    graph_def = helper.make_graph(
        nodes=nodes,
        name="lenet",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b]
    )

    # Node count: 12 (conv, relu, maxpool, conv, relu, maxpool, flatten, matmul, add, relu, matmul, add)
    # initializer count 8 (conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b)

    model_def = helper.make_model(
        graph_def,
        producer_name="testmod.py",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model_def.ir_version = 8

    base_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
    path = os.path.join(base_dir, "lenet.onnx")

    return (model_def, path)


def make_lenet_input():
    rng = np.random.default_rng(43)
    return rng.standard_normal((1,1,28,28)).astype(np.float32)


def verify_lenet(model):
    onnx.checker.check_model(model)

    ops = [n.op_type for n in model.graph.node]

    print("LeNet op types:", ops)

    assert set(ops) <= {"Conv", "Relu", "MaxPool", "Flatten", "MatMul", "Add"}, "unexpected op (Gemm?)"
    assert "Gemm" not in ops
    assert len(model.graph.node) == 12
    assert len(model.graph.initializer) == 8
    assert model.opset_import[0].version == 13


def main():
    model, model_path = build_mlp()
    verify_mlp(model)
    onnx.save(model, model_path)

    bin_path = os.path.join(os.path.dirname(model_path), "mlp_input.bin")
    make_input().tofile(bin_path)
    assert os.path.getsize(bin_path) == 16
    print("wrote", os.path.basename(model_path), "and", os.path.basename(bin_path))

    model, model_path = build_lenet()
    verify_lenet(model)
    onnx.save(model, model_path)

    bin_path = os.path.join(os.path.dirname(model_path), "lenet_input.bin")
    make_lenet_input().tofile(bin_path)
    assert os.path.getsize(bin_path) == 3136
    print("wrote", os.path.basename(model_path), "and", os.path.basename(bin_path))


if __name__ == "__main__":
    main()