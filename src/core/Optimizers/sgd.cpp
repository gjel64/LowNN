#include "core/Optimizers/sgd.hpp"

SGD::SGD(float learning_rate, const std::vector<std::shared_ptr<Tensor>>& parameters, bool maximize) 
    : Optimizer(learning_rate, parameters), _maximize(maximize) {}

void SGD::step()
{
    for (const auto& param : parameters()) {
        if (param->require_grad()) {
            // imply that paramters are leaf tensors (no stride / offset)
            std::shared_ptr<float[]> data = param->data();
            std::shared_ptr<float[]> grad = param->gradp();
            std::size_t size = param->size();
            for (std::size_t i = 0; i < size; i++) {
                if (_maximize) {
                    data[i] += _learning_rate * grad[i];
                } 
                else {
                    data[i] -= _learning_rate * grad[i];
                }
            }
        }
        else {
            throw std::runtime_error("SGD step : Trying to update a parameter that does not require grad");
        }
    }
}