#include "onnxcc/ir/attribute.h"
#include "onnxcc/ir/node.h"
#include <type_traits>    
#include <utility>
#include <stdexcept>


// Internal helpers for error messages. The anonymous namespace gives them internal linkage so they stay private to this file and can't collide with other translation units.
namespace{

// Helper to identify the type of the template T.
template <typename T>
std::string_view type_name()
{

    // std::is_same_v is a compile time boolean variable which tells if two types are same or not.
    if constexpr (std::is_same_v<T, std::int64_t>)
        return "int64";
    else if constexpr (std::is_same_v<T, float>)
        return "float";
    else if constexpr (std::is_same_v<T, std::string>)
        return "string";
    else if constexpr (std::is_same_v<T, std::vector<std::int64_t>>)
        return "vector<int64>";
    else if constexpr (std::is_same_v<T, std::vector<float>>)
        return "vector<float>";
    else if constexpr (std::is_same_v<T, onnxcc::Tensor>)
        return "Tensor";
    else
        return "unknown";
}

// Helper to identify the type name of the attribute.
std::string_view held_name(const onnxcc::Attribute &attr)
{
    // std::holds_alternative checks if a variant currently contains a value of a particular type.
    if (std::holds_alternative<std::int64_t>(attr))
        return type_name<std::int64_t>();
    if (std::holds_alternative<float>(attr))
        return type_name<float>();
    if (std::holds_alternative<std::string>(attr))
        return type_name<std::string>();
    if (std::holds_alternative<std::vector<std::int64_t>>(attr))
        return type_name<std::vector<std::int64_t>>();
    if (std::holds_alternative<std::vector<float>>(attr))
        return type_name<std::vector<float>>();
    if (std::holds_alternative<onnxcc::Tensor>(attr))
        return type_name<onnxcc::Tensor>();

    return "unknown";
}
} //namespace

template <typename T>
std::optional<T> onnxcc::get_attr(const onnxcc::Node &node, std::string_view name)
{
    // Unordered_map .find wants std::string so we convert name into it.
    auto value = node.attributes.find(std::string(name));

    // Returns nullopt when attribute not found as per requirement
    if (value == node.attributes.end())
        return std::nullopt;

    // std::get_if<T> is a non-throwing check: returns a pointer to the T if the variant holds one, otherwise nullptr.
    // Must copy, not move: 'it' points into the Node's attribute map, and node is const&.
    if (const T *it = std::get_if<T>(&value->second))
        return *it;

    // Type mismatch runtime error: clearly reports node name, op_type, attribute name and both types.
    throw std::runtime_error(
        "Node '" + node.name +
        "' (op_type '" + node.op_type +
        "'): attribute '" + std::string(name) +
        "' has type " + std::string(held_name(value->second)) +
        ", but was requested as " + std::string(type_name<T>()));
}

template <typename T>
T onnxcc::get_attr_or(const onnxcc::Node &node, std::string_view name, T fallback)
{
    auto value = onnxcc::get_attr<T>(node, name);

    // Returns fallback if the attribute is absent.
    // std::move is needed because the ternary yields an lvalue, which would not be implicitly moved on return and would be silently copied.
    return value ? std::move(*value) : std::move(fallback);
}

template <typename T>
T onnxcc::get_attr_required(const onnxcc::Node &node, std::string_view name)
{
    auto value = onnxcc::get_attr<T>(node, name);
    // Throws error if attribute is absent.
    if (value)
    {
        return std::move(*value);
    }
    throw std::runtime_error("Node '" + node.name + "' (op_type '" + node.op_type + "'): required attribute '" + std::string(name) + "' is missing");
}

// Explicitly instantiate all six supported attribute types.
#define INSTANTIATE_ATTR(T) \
    template std::optional<T> onnxcc::get_attr<T>( \
        const onnxcc::Node&, std::string_view); \
    template T onnxcc::get_attr_or<T>( \
        const onnxcc::Node&, std::string_view, T); \
    template T onnxcc::get_attr_required<T>( \
        const onnxcc::Node&, std::string_view);

INSTANTIATE_ATTR(std::int64_t)
INSTANTIATE_ATTR(float)
INSTANTIATE_ATTR(std::string)
INSTANTIATE_ATTR(std::vector<std::int64_t>)
INSTANTIATE_ATTR(std::vector<float>)
INSTANTIATE_ATTR(onnxcc::Tensor)

#undef INSTANTIATE_ATTR