#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <variant>

#include "onnxcc/ir/attribute.h"
#include "onnxcc/ir/node.h"

using namespace onnxcc;

namespace {

Node make_node()
{
    Node n;
    n.name = "conv1";
    n.op_type = "Conv";
    n.attributes.emplace("group", Attribute{std::int64_t{2}});
    n.attributes.emplace("epsilon", Attribute{1e-5f});
    n.attributes.emplace("mode", Attribute{std::string("NOTSET")});
    n.attributes.emplace("strides", Attribute{std::vector<std::int64_t>{1, 2}});
    n.attributes.emplace("scales", Attribute{std::vector<float>{0.5f, 1.5f}});
    return n;
}

// Runs f, returns the runtime_error message ("" if nothing was thrown)
template <typename F>
std::string error_of(F f)
{
    try { f(); }
    catch (const std::runtime_error &e) { return e.what(); }
    return "";
}

} // namespace

// ---------- get_attr ----------

TEST(GetAttr, ReturnsValueForEachType)
{
    Node n = make_node();
    EXPECT_EQ(get_attr<std::int64_t>(n, "group"), 2);
    EXPECT_FLOAT_EQ(*get_attr<float>(n, "epsilon"), 1e-5f);
    EXPECT_EQ(get_attr<std::string>(n, "mode"), "NOTSET");
    EXPECT_EQ(*get_attr<std::vector<std::int64_t>>(n, "strides"),
              (std::vector<std::int64_t>{1, 2}));
    EXPECT_EQ(*get_attr<std::vector<float>>(n, "scales"),
              (std::vector<float>{0.5f, 1.5f}));
}

TEST(GetAttr, AbsentReturnsNullopt)
{
    Node n = make_node();
    EXPECT_FALSE(get_attr<std::int64_t>(n, "missing").has_value());
    EXPECT_FALSE(get_attr<std::string>(n, "missing").has_value());
}

TEST(GetAttr, EmptyNodeReturnsNullopt)
{
    Node n;
    EXPECT_FALSE(get_attr<float>(n, "anything").has_value());
}

TEST(GetAttr, NameLookupIsCaseSensitive)
{
    Node n = make_node();
    EXPECT_FALSE(get_attr<std::int64_t>(n, "Group").has_value());
}

TEST(GetAttr, TypeMismatchThrowsRuntimeError)
{
    Node n = make_node();
    EXPECT_THROW(get_attr<float>(n, "group"), std::runtime_error);
}

TEST(GetAttr, TypeMismatchDoesNotLeakBadVariantAccess)
{
    Node n = make_node();
    EXPECT_NO_THROW({
        try { get_attr<float>(n, "group"); }
        catch (const std::runtime_error &) {}
    });
}

TEST(GetAttr, MismatchMessageNamesEverything)
{
    Node n = make_node();
    std::string msg = error_of([&] { get_attr<float>(n, "group"); });
    EXPECT_NE(msg.find("conv1"), std::string::npos);    // node name
    EXPECT_NE(msg.find("Conv"), std::string::npos);     // op_type
    EXPECT_NE(msg.find("group"), std::string::npos);    // attribute name
    EXPECT_NE(msg.find("int64"), std::string::npos);    // held type
    EXPECT_NE(msg.find("float"), std::string::npos);    // requested type
}

TEST(GetAttr, VectorVsScalarMismatchIsDistinguished)
{
    Node n = make_node();
    std::string msg = error_of([&] { get_attr<std::int64_t>(n, "strides"); });
    EXPECT_NE(msg.find("vector<int64>"), std::string::npos);
}

TEST(GetAttr, DoesNotModifyNode)
{
    Node n = make_node();
    get_attr<std::vector<std::int64_t>>(n, "strides");
    // Value must still be there after a read (copy, not move)
    EXPECT_EQ(std::get<std::vector<std::int64_t>>(n.attributes.at("strides")).size(), 2u);
}

// ---------- get_attr_or ----------

TEST(GetAttrOr, ReturnsValueWhenPresent)
{
    Node n = make_node();
    EXPECT_EQ(get_attr_or<std::int64_t>(n, "group", 99), 2);
}

TEST(GetAttrOr, ReturnsFallbackWhenAbsent)
{
    Node n = make_node();
    EXPECT_EQ(get_attr_or<std::int64_t>(n, "missing", 99), 99);
    EXPECT_EQ(get_attr_or<std::string>(n, "missing", "dflt"), "dflt");
    EXPECT_EQ(get_attr_or<std::vector<std::int64_t>>(n, "missing", {7, 8}),
              (std::vector<std::int64_t>{7, 8}));
}

TEST(GetAttrOr, StillThrowsOnTypeMismatch)
{
    Node n = make_node();
    // Fallback must NOT mask a wrong-type attribute
    EXPECT_THROW(get_attr_or<float>(n, "group", 1.0f), std::runtime_error);
}

// ---------- get_attr_required ----------

TEST(GetAttrRequired, ReturnsValueWhenPresent)
{
    Node n = make_node();
    EXPECT_EQ(get_attr_required<std::string>(n, "mode"), "NOTSET");
}

TEST(GetAttrRequired, ThrowsWhenAbsentWithUsefulMessage)
{
    Node n = make_node();
    std::string msg = error_of([&] { get_attr_required<std::int64_t>(n, "kernel_shape"); });
    EXPECT_NE(msg.find("conv1"), std::string::npos);
    EXPECT_NE(msg.find("Conv"), std::string::npos);
    EXPECT_NE(msg.find("kernel_shape"), std::string::npos);
    EXPECT_NE(msg.find("missing"), std::string::npos);
}

TEST(GetAttrRequired, ThrowsOnTypeMismatch)
{
    Node n = make_node();
    EXPECT_THROW(get_attr_required<float>(n, "group"), std::runtime_error);
}