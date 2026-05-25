#include <gtest/gtest.h>
#include <vector>
#include "core/Math/vecUtils.hpp"

TEST(VecUtilsTest, FlattenInto) 
{
    std::vector<std::vector<float>> nestedVec = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    std::vector<float> flatVec;
    VecUtils::flatten_into(nestedVec, flatVec);
    EXPECT_EQ(flatVec.size(), 4);
    EXPECT_FLOAT_EQ(flatVec[0], 1.0f);
    EXPECT_FLOAT_EQ(flatVec[1], 2.0f);
    EXPECT_FLOAT_EQ(flatVec[2], 3.0f);
    EXPECT_FLOAT_EQ(flatVec[3], 4.0f);
}

TEST(VecUtilsTest, Broadcast) 
{
    std::vector<std::size_t> shape1 = {4, 3};
    std::vector<std::size_t> shape2 = {1, 3};
    std::vector<std::size_t> expected_shape = {4, 3};
    EXPECT_EQ(VecUtils::broadcast(shape1, shape2), expected_shape);

    shape1 = {5, 4, 3};
    shape2 = {4, 3};
    expected_shape = {5, 4, 3};
    EXPECT_EQ(VecUtils::broadcast(shape1, shape2), expected_shape);

    shape1 = {2, 3};
    shape2 = {3};
    expected_shape = {2, 3};
    EXPECT_EQ(VecUtils::broadcast(shape1, shape2), expected_shape);

    shape1 = {2, 3};
    shape2 = {4, 5};
    EXPECT_THROW(VecUtils::broadcast(shape1, shape2), std::invalid_argument);

    shape1 = {3};
    shape2 = {1, 3};
    expected_shape = {1, 3};
    EXPECT_EQ(VecUtils::broadcast(shape1, shape2), expected_shape);

    shape1 = {3};
    shape2 = {};
    expected_shape = {3};
    EXPECT_EQ(VecUtils::broadcast(shape1, shape2), expected_shape);

    shape1 = {};
    shape2 = {};
    expected_shape = {};
    EXPECT_EQ(VecUtils::broadcast(shape1, shape2), expected_shape);

    shape1 = {5, 7, 3, 4, 12, 4};
    shape2 = {1, 1, 3, 1, 12, 1};
    expected_shape = {5, 7, 3, 4, 12, 4};
    EXPECT_EQ(VecUtils::broadcast(shape1, shape2), expected_shape);
}