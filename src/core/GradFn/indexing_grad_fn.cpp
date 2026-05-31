#include "core/GradFn/indexing_grad_fn.hpp"
#include "core/tensor.hpp"

IndexingGradFn::IndexingGradFn(std::vector<std::weak_ptr<Tensor>> inputs, std::weak_ptr<Tensor> output)
    : GradFn("indexing", inputs, output)
{
}

void IndexingGradFn::backward(
    const std::shared_ptr<Tensor>& out,
    const std::shared_ptr<Tensor>& out_grad
){
    auto a = inputs[0].lock();
    if (!a) {
        throw std::runtime_error("IndexingGradFn: input tensor no longer exist");
    }

    a->ensure_grad_allocated();
    std::shared_ptr<float[]> a_grad_data = a->gradp();
    const std::size_t input_offset = out->offset() - a->offset();
    for (std::size_t i = 0; i < out->size(); i++) {
        a_grad_data.get()[i + input_offset] += out_grad->data().get()[i];
    }
}