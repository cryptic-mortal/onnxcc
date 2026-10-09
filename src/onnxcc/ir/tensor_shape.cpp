#include "tensor_shape.h"
#include "datatype.h"
#include <limits>

std::size_t onnxcc::TensorShape::num_elements() const
{
    if (is_dynamic())
    {
        throw std::logic_error("cannot determine the number of elements of a dynamic tensor shape");
    }

    // Result is initialized to 1. If dims is empty, the loop never runs and the result would be 1 which is correct for scalar
    std::size_t result = 1;
    // Numeric limits sets max size according to the system.
    std::size_t maxi = std::numeric_limits<std::size_t>::max();

    for (const std::int64_t &dim : dims)
    {
        // We have to check everytime if result*dim > max range and that dim is not 0.
        if (dim != 0 && (maxi / dim) < result)
        {
            throw std::overflow_error("number of tensor elements exceeds the representable range");
        }
        result *= dim;
    }
    return result;
}

std::size_t onnxcc::TensorShape::num_bytes(onnxcc::DataType dt) const
{
    std::size_t elements = num_elements();
    std::size_t data_size = dtype_size(dt);
    if ((std::numeric_limits<std::size_t>::max()) / data_size < elements)
    {
        throw std::overflow_error("number of tensor bytes exceeds the representable range");
    }
    return data_size * elements;
}

std::string onnxcc::TensorShape::to_string() const
{
    std::string result = "[";
    for (const std::int64_t &dim : dims)
    {
        // If string has no element then we do not put a comma.
        if (result != "[")
        {
            result += ", ";
        }
        if (dim < 0)
        {
            result += "?";
        }
        else
        {
            result += std::to_string(dim);
        }
    }
    result += "]";
    return result;
}

bool onnxcc::TensorShape::is_dynamic() const noexcept
{
    for (const std::int64_t &dim : dims)
    {
        // If dim < 0, that means that TensorShape is dynamic.
        if (dim < 0)
        {
            return true;
        }
    }
    return false;
}