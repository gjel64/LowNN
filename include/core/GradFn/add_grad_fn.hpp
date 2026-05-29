#pragma once
#include "core/grad_fn.hpp"
#include "core/Math/vecUtils.hpp"

class AddGradFn : public GradFn {
public:
    AddGradFn(std::vector<std::weak_ptr<Tensor>> inputs, std::weak_ptr<Tensor> output);
    
    std::vector<array> backward(
        const std::shared_ptr<Tensor>& out,
        const std::shared_ptr<Tensor>& out_grad
    ) override;
};
