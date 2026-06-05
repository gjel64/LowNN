#pragma once 
#include "core/Module/module.hpp"

class Sequential : public Module {
private:
    std::vector<std::shared_ptr<Module>> modules;
public:
    Sequential(std::vector<std::shared_ptr<Module>> modules);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void add(std::shared_ptr<Module> module);
    std::shared_ptr<Module> operator[](std::size_t index) const;
};