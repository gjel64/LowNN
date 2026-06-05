#include "core/GradFn/exp_grad_fn.hpp"
#include "core/tensor.hpp"

ExpGradFn::ExpGradFn(std::vector<std::weak_ptr<Tensor>> inputs, std::weak_ptr<Tensor> output)
    : GradFn("exp", inputs, output)
{
}

void ExpGradFn::backward(
    const std::shared_ptr<Tensor>& out,
    const std::shared_ptr<Tensor>& out_grad
){
    auto a = inputs[0].lock();
    if (!a) {
        throw std::runtime_error("ExpGradFn: input tensor no longer exist");
    }
    std::shared_ptr<float[]> a_grad_data = (a->gradp() == nullptr) ? std::make_shared<float[]>(a->size()) : a->gradp();
    for (std::size_t i = 0; i < out->size(); i++) {
        a_grad_data.get()[(i % a->size()) + a->offset()] += out->data().get()[i + out->offset()] * out_grad->data().get()[i + out_grad->offset()];
    }
}