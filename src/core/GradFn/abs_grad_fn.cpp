#include "core/GradFn/abs_grad_fn.hpp"
#include "core/tensor.hpp"

AbsGradFn::AbsGradFn(std::vector<std::weak_ptr<Tensor>> inputs, std::weak_ptr<Tensor> output)
    : GradFn("abs", inputs, output)
{
}

void AbsGradFn::backward(
    const std::shared_ptr<Tensor>& out,
    const std::shared_ptr<Tensor>& out_grad
)
{
    auto a = inputs[0].lock();
    if (!a) {
        throw std::runtime_error("AbsGradFn: input tensor no longer exist");
    }
    
    std::shared_ptr<float[]> a_grad_data = a->gradp() ? a->gradp() : std::make_shared<float[]>(a->size());

    for (std::size_t i = 0; i < out->size(); i++) {
        if (a->data().get()[(i % a->size()) + a->offset()] > 0) {
            a_grad_data.get()[(i % a->size()) + a->offset()] += out_grad->data().get()[i + out_grad->offset()];
        }
        else if (a->data().get()[(i % a->size()) + a->offset()] < 0) {
            a_grad_data.get()[(i % a->size()) + a->offset()] -= out_grad->data().get()[i + out_grad->offset()];
        }
    }
}