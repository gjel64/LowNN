#pragma once
#include "core/grad_fn.hpp"
#include "core/Math/vecUtils.hpp"

class IndexingGradFn : public GradFn {
public:
    IndexingGradFn(std::vector<std::weak_ptr<Tensor>> inputs, std::weak_ptr<Tensor> output);
    
    void backward(
        const std::shared_ptr<Tensor>& out,
        const std::shared_ptr<Tensor>& out_grad
    ) override;
};
