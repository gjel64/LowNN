#include "core/GradFn/pow_grad_fn.hpp"
#include "core/tensor.hpp"

PowGradFn::PowGradFn(std::vector<std::weak_ptr<Tensor>> inputs, std::weak_ptr<Tensor> output)
    : GradFn("pow", inputs, output)
{
}

void PowGradFn::backward(
    const std::shared_ptr<Tensor>& out,
    const std::shared_ptr<Tensor>& out_grad
){
    auto a = inputs[0].lock();
    auto b = inputs[1].lock();
    if (!a || !b) {
        throw std::runtime_error("PowGradFn: input tensors no longer exist");
    }
    std::shared_ptr<float[]> a_grad_data = (a->gradp() == nullptr) ? std::make_shared<float[]>(a->size()) : a->gradp();
    std::shared_ptr<float[]> b_grad_data = (b->gradp() == nullptr) ? std::make_shared<float[]>(b->size()) : b->gradp();

    if (b->size() == 1) {
        float b_scalar = b->data().get()[b->offset()];
        for (std::size_t i = 0; i < out->size(); i++) {
            a_grad_data.get()[i % a->size()] += out_grad->data().get()[i + out_grad->offset()] * b_scalar * std::pow(a->data().get()[i % a->size()], b_scalar - 1);
            b_grad_data.get()[0] += out_grad->data().get()[i + out_grad->offset()] * std::pow(a->data().get()[i % a->size()], b_scalar) * std::log(a->data().get()[i % a->size()]);
        }
    } 
    else {
        throw std::runtime_error("PowGradFn: backward only implemented for scalar exponent");
        for (std::size_t i = 0; i < out->size(); i++) {
            a_grad_data.get()[i % a->size()] += out_grad->data().get()[i + out_grad->offset()] * b->data().get()[i % b->size()] * std::pow(a->data().get()[i % a->size()], b->data().get()[i % b->size()] - 1);
            b_grad_data.get()[i % b->size()] += out_grad->data().get()[i + out_grad->offset()] * std::pow(a->data().get()[i % a->size()], b->data().get()[i % b->size()]) * std::log(a->data().get()[i % a->size()]);
        }
    }
}
