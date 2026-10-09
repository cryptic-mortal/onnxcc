#include "datatype.h"

std::string_view onnxcc::dtype_name(onnxcc::DataType dt){
    switch(dt) {
        case onnxcc::DataType::FLOAT32: return "FLOAT32";
        case onnxcc::DataType::INT32: return "INT32";
        case onnxcc::DataType::INT64: return "INT64";
        case onnxcc::DataType::BOOL: return "BOOL";
        case onnxcc::DataType::FLOAT64: return "FLOAT64";
        case onnxcc::DataType::UNDEFINED: break;
    }
    // does not require caller to check status
    // if the datatype is not defined/undefined it is programmer's error 
    throw std::logic_error("dtype_name called on UNDEFINED or unknown datatype");
}