#include <gtest/gtest.h>
#include "core/tensor.h"
#include <iostream>
#include <memory>

TEST(TensorTest, Creation) {
    std::cout << "TEST: Creation" << std::endl;
    std::cout << "  t_0" << std::endl;
    std::shared_ptr<Tensor> t_0 = std::make_shared<Tensor>(2.0f);
    EXPECT_EQ(t_0->item(), 2.0f);
    EXPECT_EQ( (*t_0)[0]->item(), 2.0f);
    EXPECT_EQ(t_0->shape().size(), 1);
    EXPECT_EQ(t_0->strides().size(), 1);
    EXPECT_EQ(t_0->size(), 1);

    std::cout << "  t_1" << std::endl;
    std::shared_ptr<Tensor> t_1 = std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f, 3.0f});
    EXPECT_THROW(t_1->item(), std::invalid_argument);
    EXPECT_EQ( (*t_1)[1]->item(), 2.0f);
    EXPECT_EQ(t_1->shape().size(), 1);
    EXPECT_EQ(t_1->shape()[0], 3);
    EXPECT_EQ(t_1->strides().size(), 1);
    EXPECT_EQ(t_1->strides()[0], 1);
    EXPECT_EQ(t_1->size(), 3);


    std::cout << "  t_2" << std::endl;
    std::shared_ptr<Tensor> t_2 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    EXPECT_THROW(t_2->item(), std::invalid_argument);
    EXPECT_EQ( (*(*t_2)[1])[1]->item(), 4.0f);
    EXPECT_EQ( ((*t_2)[std::vector<int>{1, 1}])->item(), 4.0f);
    EXPECT_EQ(t_2->shape().size(), 2);
    EXPECT_EQ(t_2->shape()[0], 2);
    EXPECT_EQ(t_2->shape()[1], 2);
    EXPECT_EQ(t_2->strides().size(), 2);
    EXPECT_EQ(t_2->strides()[0], 2);
    EXPECT_EQ(t_2->strides()[1], 1);
    EXPECT_EQ(t_2->size(), 4);
}