#include "core/Module/Modules/tanh.hpp"

Tanh::Tanh() {
}

std::shared_ptr<Tensor> Tanh::forward(std::shared_ptr<Tensor> input) {
    return input->tanh();
}

std::vector<std::shared_ptr<Tensor>> Tanh::parameters() {
    return std::vector<std::shared_ptr<Tensor>>{};
}