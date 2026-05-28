#include "core/GradFn/add_grad_fn.hpp"
#include "core/tensor.hpp"

AddGradFn::AddGradFn(std::vector<std::weak_ptr<Tensor>> inputs, std::weak_ptr<Tensor> output)
    : GradFn("add", inputs, output)
{
}

std::vector<std::shared_ptr<Tensor>> AddGradFn::backward(
    const std::shared_ptr<Tensor>& out,
    const std::shared_ptr<Tensor>& out_grad
){
    //no broadcasting support yet
    auto a = inputs[0].lock();
    auto b = inputs[1].lock();
    if (!a || !b) {
        throw std::runtime_error("AddGradFn: input expired");
    }
    if (a->shape() != out->shape() || b->shape() != out->shape()) {
        throw std::runtime_error("AddGradFn: broadcasting not supported in backward");
    }

    // return views of out_grad for both inputs (no copy)
    auto g_a = std::make_shared<Tensor>(out_grad->data(), a->shape(), out_grad->offset(), a->size(), false);
    auto g_b = std::make_shared<Tensor>(out_grad->data(), b->shape(), out_grad->offset(), b->size(), false);

    return {g_a, g_b};
}
