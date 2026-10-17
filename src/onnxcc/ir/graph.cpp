#include "onnxcc/ir/graph.h"

#include <utility>

namespace onnxcc {

namespace {

void reset(Graph& g) noexcept {
    g.name.clear();
    g.ir_version = 0;
    g.opset_version = 0;
    g.inputs.clear();
    g.outputs.clear();
    g.initializers.clear();
    g.value_info.clear();
    g.nodes.clear();
}
// when we move a graph, we steal its content, but it works differently for different members like vector is moved and leave old 
// vector empty, but plain num like ir_version is copied and old graph still have that value, and we have to leave source graph empty,
// so the function reset is used to reset the source graph after move.

const Tensor* find_by_name(const std::vector<Tensor>& tensors, std::string_view name) {
    for (const Tensor& t : tensors) {
        if (t.name == name) {
            return &t;
        }
    }
    return nullptr;
}

}  // namespace

Graph::Graph(Graph&& other) noexcept
    : name(std::move(other.name)),
      ir_version(other.ir_version),
      opset_version(other.opset_version),
      inputs(std::move(other.inputs)),
      outputs(std::move(other.outputs)),
      initializers(std::move(other.initializers)),
      value_info(std::move(other.value_info)),
      nodes(std::move(other.nodes)) {
    reset(other);
}
// this is the move constructor, it moves the content of other graph to a newly built graph that is empty, and then reset the other graph to empty.

Graph& Graph::operator=(Graph&& other) noexcept {
    if (this != &other) {
        name = std::move(other.name);
        ir_version = other.ir_version;
        opset_version = other.opset_version;
        inputs = std::move(other.inputs);
        outputs = std::move(other.outputs);
        initializers = std::move(other.initializers);
        value_info = std::move(other.value_info);
        nodes = std::move(other.nodes);
        reset(other);
    }
    return *this;
}
// this is the move assignment operator, it moves the content of other graph to already existing graph by releasing its data and moving the 
// content, and then reset the other graph to empty.

const Tensor* Graph::find_initializer(std::string_view tensor_name) const {
    return find_by_name(initializers, tensor_name);
}

const Tensor* Graph::find_value_info(std::string_view tensor_name) const {
    return find_by_name(value_info, tensor_name);
}

}  // namespace onnxcc