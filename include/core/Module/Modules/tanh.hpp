#pragma once 
#include "core/Module/module.hpp"

class Tanh : public Module {
public:
    Tanh();
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
};