#include "core/GradFn/mul_grad_fn.hpp"
#include "core/tensor.hpp"

MulGradFn::MulGradFn(std::vector<std::weak_ptr<Tensor>> inputs, std::weak_ptr<Tensor> output)
    : GradFn("mul", inputs, output)
{
}

void MulGradFn::backward(
    const std::shared_ptr<Tensor>& out,
    const std::shared_ptr<Tensor>& out_grad
){
    auto a = inputs[0].lock();
    auto b = inputs[1].lock();
    if (!a || !b) {
        throw std::runtime_error("MulGradFn: input tensors no longer exist");
    }
    std::shared_ptr<float[]> a_grad_data = a->gradp() ? a->gradp() : std::make_shared<float[]>(a->size());
    std::shared_ptr<float[]> b_grad_data = b->gradp() ? b->gradp() : std::make_shared<float[]>(b->size());
    
    for (std::size_t i = 0; i < out->size(); i++) {
        a_grad_data.get()[i % a->size()] += out_grad->data().get()[i + out_grad->offset()] * b->data().get()[i % b->size()];
        b_grad_data.get()[i % b->size()] += out_grad->data().get()[i + out_grad->offset()] * a->data().get()[i % a->size()];
    }
}
