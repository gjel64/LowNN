#include "core/optimizer.hpp"

void Optimizer::zero_grad()
{
    for (std::shared_ptr<Tensor> param : _parameters) {
        if (param->gradp()) {
            float* grad_data = param->gradp().get();
            std::size_t size = param->size();
            for (std::size_t i = 0; i < size; i++) {
                grad_data[i] = 0.0f;
            }
        }
    }
}