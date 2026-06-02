#pragma once
#include "core/grad_fn.hpp"
#include "core/Utils/vecUtils.hpp"

class AbsGradFn : public GradFn {
public:
    AbsGradFn(std::vector<std::weak_ptr<Tensor>> inputs, std::weak_ptr<Tensor> output);
    
    void backward(
        const std::shared_ptr<Tensor>& out,
        const std::shared_ptr<Tensor>& out_grad
    ) override;
};
