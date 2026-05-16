#include <gtest/gtest.h>
#include <vector>
#include "core/Math/vecUtils.h"

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