#include "core/GradFn/mean_grad_fn.hpp"
#include "core/tensor.hpp"

MeanGradFn::MeanGradFn(std::vector<std::weak_ptr<Tensor>> inputs, std::weak_ptr<Tensor> output)
    : GradFn("mean", inputs, output)
{
}

void MeanGradFn::backward(
    const std::shared_ptr<Tensor>& out,
    const std::shared_ptr<Tensor>& out_grad
)
{
    auto a = inputs[0].lock();
    if (!a) {
        throw std::runtime_error("MeanGradFn: input tensor no longer exist");
    }
    
    std::shared_ptr<float[]> a_grad_data = a->gradp() ? a->gradp() : std::make_shared<float[]>(a->size());

    for (std::size_t i = 0; i < a->size(); i++) {
        a_grad_data.get()[(i % a->size()) + a->offset()] += (1.0f / a->size()) * out_grad->data().get()[(i + out->offset()) % out->size()];
    }
}