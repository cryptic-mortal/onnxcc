#include <gtest/gtest.h>

#include "onnxcc/ir/datatype.h"

// Test for dtype_size()
TEST(DataTypeTest, DtypeSizeReturnsCorrectSizes) {
    EXPECT_EQ(onnxcc::dtype_size(onnxcc::DataType::FLOAT32), 4);
    EXPECT_EQ(onnxcc::dtype_size(onnxcc::DataType::INT32), 4);
    EXPECT_EQ(onnxcc::dtype_size(onnxcc::DataType::INT64), 8);
    EXPECT_EQ(onnxcc::dtype_size(onnxcc::DataType::BOOL), 1);
    EXPECT_EQ(onnxcc::dtype_size(onnxcc::DataType::FLOAT64), 8);
}

TEST(DataTypeTest, DtypeSizeUndefinedThrowsException) {
    EXPECT_THROW(
        onnxcc::dtype_size(onnxcc::DataType::UNDEFINED),
        std::logic_error
    );
}


// Test for dtype_name
TEST(DataTypeTest, ReturnsCorrectNames) {
    EXPECT_EQ(dtype_name(onnxcc::DataType::FLOAT32), "FLOAT32");
    EXPECT_EQ(dtype_name(onnxcc::DataType::INT32), "INT32");
    EXPECT_EQ(dtype_name(onnxcc::DataType::INT64), "INT64");
    EXPECT_EQ(dtype_name(onnxcc::DataType::BOOL), "BOOL");
    EXPECT_EQ(dtype_name(onnxcc::DataType::FLOAT64), "FLOAT64");
}

TEST(DataTypeTest, UndefinedThrowsException) {
    EXPECT_THROW(
        dtype_name(onnxcc::DataType::UNDEFINED),
        std::logic_error
    );
}