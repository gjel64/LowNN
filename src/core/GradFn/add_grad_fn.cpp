#include "core/GradFn/add_grad_fn.hpp"
#include "core/tensor.hpp"

AddGradFn::AddGradFn(std::vector<std::weak_ptr<Tensor>> inputs, std::weak_ptr<Tensor> output)
    : GradFn("add", inputs, output)
{
}

void AddGradFn::backward(
    const std::shared_ptr<Tensor>& out,
    const std::shared_ptr<Tensor>& out_grad
){
    auto a = inputs[0].lock();
    auto b = inputs[1].lock();
    if (!a || !b) {
        throw std::runtime_error("AddGradFn: input tensors no longer exist");
    }
    if (VecUtils::broadcast(b->shape(), a->shape()) != out->shape()) {
        throw std::runtime_error("AddGradFn: shapes are not compatible for broadcasting (a/b vs out) -> not usual error");
    }
    std::shared_ptr<float[]> a_grad_data = a->gradp() ? a->gradp() : std::make_shared<float[]>(a->size());
    std::shared_ptr<float[]> b_grad_data = b->gradp() ? b->gradp() : std::make_shared<float[]>(b->size());
    for (std::size_t i = 0; i < out->size(); ++i) {
        a_grad_data.get()[i % a->size()] += out_grad->data().get()[i + out_grad->offset()];
        b_grad_data.get()[i % b->size()] += out_grad->data().get()[i + out_grad->offset()];
    }
}
