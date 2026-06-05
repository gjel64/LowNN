#include <gtest/gtest.h>
#include "core/tensor.hpp"
#include <memory>
#include "core/Module/module.hpp"
#include "core/Module/Modules/linear.hpp"
#include "core/Module/Modules/tanh.hpp"

TEST(ModuleTest, Linear) 
{
    std::size_t in_features = 3;
    std::size_t out_features = 2;
    Linear linear(in_features, out_features);

    std::shared_ptr<Tensor> input = std::make_shared<Tensor>(Tensor::randn(std::vector<std::size_t>{1, in_features}));
    std::shared_ptr<Tensor> output = linear.forward(input);

    EXPECT_EQ(output->shape(), (std::vector<std::size_t>{1, out_features}));
    EXPECT_EQ(linear.parameters().size(), 2);
    EXPECT_EQ(linear.parameters()[0]->shape(), (std::vector<std::size_t>{in_features, out_features}));
    EXPECT_EQ(linear.parameters()[1]->shape(), (std::vector<std::size_t>{out_features}));

    Linear linear2(in_features, out_features);
    linear2.set_weight(std::make_shared<Tensor>(Tensor::fill(0.5f, {in_features, out_features})));
    linear2.set_bias(std::make_shared<Tensor>(Tensor::fill(1.0f, {out_features})));
    std::shared_ptr<Tensor> output2 = linear2.forward(std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f, 3.0f}}));
    EXPECT_EQ(output2->shape(), (std::vector<std::size_t>{1, out_features}));
    std::vector<float> expected_output = {1.0f*0.5f + 2.0f*0.5f + 3.0f*0.5f + 1.0f, 1.0f*0.5f + 2.0f*0.5f + 3.0f*0.5f + 1.0f};
    for (std::size_t i = 0; i < out_features; i++) {
        EXPECT_FLOAT_EQ(output2->data()[i], expected_output[i]);
    }
}

TEST(ModuleTest, Tanh) 
{
    Tanh tanh;
    std::shared_ptr<Tensor> input = std::make_shared<Tensor>(std::vector<float>{0.0f, 1.0f, -1.0f}, true);
    std::shared_ptr<Tensor> output = tanh(input);

    EXPECT_EQ(output->shape(), (std::vector<std::size_t>{3}));
    EXPECT_FLOAT_EQ(output->data()[0], 0.0f);
    EXPECT_FLOAT_EQ(output->data()[1], std::tanh(1.0f));
    EXPECT_FLOAT_EQ(output->data()[2], std::tanh(-1.0f));

    output->backward(std::make_shared<Tensor>(Tensor::fill(1.0f, {3})));

    EXPECT_FLOAT_EQ(input->gradp().get()[0], 1.0f - std::tanh(0.0f) * std::tanh(0.0f));
    EXPECT_FLOAT_EQ(input->gradp().get()[1], 1.0f - std::tanh(1.0f) * std::tanh(1.0f));
    EXPECT_FLOAT_EQ(input->gradp().get()[2], 1.0f - std::tanh(-1.0f) * std::tanh(-1.0f));

}