#pragma once
#include "core/tensor.hpp"
#include <vector>
#include <memory>

class Module {
public:
    Module();
    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) = 0;
    std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> input);
    virtual std::vector<std::shared_ptr<Tensor>> parameters() = 0;

};