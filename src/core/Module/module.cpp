#include "core/Module/module.hpp"

Module::Module() 
{}

std::shared_ptr<Tensor> Module::operator()(std::shared_ptr<Tensor> input) {
    return forward(input);
}

std::vector<std::shared_ptr<Tensor>> Module::parameters() {
    return {};
}