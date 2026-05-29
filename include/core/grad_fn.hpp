#pragma once
#include <string>
#include <memory>
#include <vector>
#include "core/Math/vecUtils.hpp"

class Tensor; // forward declaration

class GradFn {
public:
    std::string name;
    std::vector<std::weak_ptr<Tensor>> inputs;
    std::weak_ptr<Tensor> output;

    GradFn(std::string name, std::vector<std::weak_ptr<Tensor>> inputs = {}, std::weak_ptr<Tensor> output = {});

    // Return a vector of gradients matching inputs order. Each gradient is a Tensor (view or new).
    virtual std::vector<array> backward(
        const std::shared_ptr<Tensor>& out,
        const std::shared_ptr<Tensor>& out_grad
    ) = 0;
};