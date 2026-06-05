#include "core/Module/Modules/sequential.hpp"

Sequential::Sequential(std::vector<std::shared_ptr<Module>> modules) : modules(modules) {}

std::shared_ptr<Tensor> Sequential::forward(std::shared_ptr<Tensor> input) {
    std::shared_ptr<Tensor> output = input;
    for (const auto& module : modules) {
        output = module->forward(output);
    }
    return output;
}

std::vector<std::shared_ptr<Tensor>> Sequential::parameters() {
    std::vector<std::shared_ptr<Tensor>> params;
    for (const auto& module : modules) {
        auto module_params = module->parameters();
        params.insert(params.end(), module_params.begin(), module_params.end());
    }
    return params;
}

void Sequential::add(std::shared_ptr<Module> module) {
    modules.push_back(module);
}


std::shared_ptr<Module> Sequential::operator[](std::size_t index) const {
    if (index >= modules.size()) {
        throw std::out_of_range("Index out of range in Sequential module");
    }
    return modules[index];
}