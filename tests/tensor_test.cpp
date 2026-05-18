#include <gtest/gtest.h>
#include "core/tensor.hpp"
#include <iostream>
#include <memory>

TEST(TensorTest, Creation) 
{
    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(std::vector<float>{3.14f, 2.2f});
    EXPECT_EQ(t1->data().size(), 2);
    EXPECT_EQ(t1->data().at(0), 3.14f);
    EXPECT_EQ(t1->shape(), std::vector<std::size_t>{2});
    EXPECT_EQ(t1->strides().at(0), 1);
    EXPECT_EQ(t1->strides().size(), 1);
    EXPECT_EQ(t1->size(), 2);

    std::shared_ptr<Tensor> t2 = std::make_shared<Tensor>(3.14f);
    EXPECT_EQ(t2->data(), std::vector<float>{3.14f});
    EXPECT_EQ(t2->shape(), std::vector<std::size_t>{1});
    EXPECT_EQ(t2->strides(), std::vector<std::size_t>{});
    EXPECT_EQ(t2->size(), 1);

    EXPECT_THROW( 
        std::shared_ptr<Tensor> t3 = std::make_shared<Tensor>(3), 
        std::invalid_argument
    );

    std::shared_ptr<Tensor> t4 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    std::vector<float> d = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<std::size_t> s = std::vector<std::size_t>{2, 2};
    std::vector<std::size_t> st = std::vector<std::size_t>{2, 1};
    EXPECT_EQ(t4->data(), d);
    EXPECT_EQ(t4->shape(), s);
    EXPECT_EQ(t4->strides(), st);
    EXPECT_EQ(t4->size(), 4);

    std::shared_ptr<Tensor> t5 = std::make_shared<Tensor>(std::vector<std::vector<std::vector<float>>>{{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}});
    std::vector<float> d5 = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<std::size_t> s5 = std::vector<std::size_t>{2, 2, 2};
    std::vector<std::size_t> st5 = std::vector<std::size_t>{4, 2, 1};
    EXPECT_EQ(t5->data(), d5);
    EXPECT_EQ(t5->shape(), s5);
    EXPECT_EQ(t5->strides(), st5);
    EXPECT_EQ(t5->size(), 8);

    EXPECT_THROW( 
        std::shared_ptr<Tensor> t6 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f}, {2.0f}}), 
        std::invalid_argument
    );
}

TEST(TensorTest, BasicMethods)
{
    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(std::vector<float>{3.14f});
    EXPECT_FLOAT_EQ(t1->item(), 3.14f);
    EXPECT_STREQ(t1->to_string().c_str(), "Tensor ( data: [3.140000] )");

    std::shared_ptr<Tensor> t2 = std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f});
    EXPECT_THROW(t2->item(), std::runtime_error);
    EXPECT_STREQ(t2->to_string().c_str(), "Tensor ( data: [1.000000, 2.000000] )");

} 