#include "core/GradFn/add_grad_fn.hpp"
#include "core/tensor.hpp"

AddGradFn::AddGradFn(std::vector<std::weak_ptr<Tensor>> inputs, std::weak_ptr<Tensor> output)
    : GradFn("add", inputs, output)
{
}

std::vector<array> AddGradFn::backward(
    const std::shared_ptr<Tensor>& out,
    const std::shared_ptr<Tensor>& out_grad
){
    auto a = inputs[0].lock();
    auto b = inputs[1].lock();
    if (!a || !b) {
        throw std::runtime_error("AddGradFn: input tensors no longer exist");
    }
    if (VecUtils::broadcast(b->shape(), a->shape()) != out->shape()) {
        throw std::runtime_error("AddGradFn: shapes are not compatible for broadcasting (a/b vs out)");
    }

    return {array{out_grad->data(), a->size()}, array{out_grad->data(),  b->size()}};
}
