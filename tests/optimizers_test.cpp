#include <gtest/gtest.h>
#include "core/tensor.hpp"
#include <memory>
#include "core/Module/module.hpp"
#include "core/Module/Modules/linear.hpp"
#include "core/Module/Modules/tanh.hpp"
#include "core/Module/Modules/sequential.hpp"
#include "core/optimizer.hpp"
#include "core/Optimizers/sgd.hpp"

TEST(Optimizer, SGD)
{
    Sequential seq = Sequential({
        std::make_shared<Linear>(3, 4),
        std::make_shared<Tanh>(),
        std::make_shared<Linear>(4, 2)
    });

    dynamic_pointer_cast<Linear>(seq[0])->set_weight(std::make_shared<Tensor>(Tensor::fill(0.5f, {3, 4})));
    dynamic_pointer_cast<Linear>(seq[0])->set_bias(std::make_shared<Tensor>(Tensor::fill(1.0f, {4})));
    dynamic_pointer_cast<Linear>(seq[2])->set_weight(std::make_shared<Tensor>(Tensor::fill(0.5f, {4, 2})));
    dynamic_pointer_cast<Linear>(seq[2])->set_bias(std::make_shared<Tensor>(Tensor::fill(1.0f, {2})));

    std::shared_ptr<Tensor> input = std::make_shared<Tensor>(Tensor::fill(2.0f, {1, 3}));
    input->set_require_grad(true);

    std::shared_ptr<Tensor> output = seq(input);

    EXPECT_EQ(output->shape(), (std::vector<std::size_t>{1, 2}));
    EXPECT_EQ(seq.parameters().size(), 4);

    output->backward(std::make_shared<Tensor>(Tensor::fill(1.0f, output->shape())));
    EXPECT_NEAR(input->gradp().get()[0], 0.002681786f, 1e-6f);
    EXPECT_NEAR(input->gradp().get()[1], 0.002681786f, 1e-6f);
    EXPECT_NEAR(input->gradp().get()[2], 0.002681786f, 1e-6f);

    SGD optimizer(0.1f, seq.parameters());
    optimizer.step();
    EXPECT_FLOAT_EQ(
        dynamic_pointer_cast<Linear>(seq[0])->parameters()[0]->data().get()[0], 
        0.5f - (0.1f * dynamic_pointer_cast<Linear>(seq[0])->parameters()[0]->gradp().get()[0])
    );
}