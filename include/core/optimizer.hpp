#pragma once
#include <vector>
#include <memory>
#include "core/tensor.hpp"

class Optimizer {
protected:
    float _learning_rate;
    std::vector<std::shared_ptr<Tensor>> _parameters;
public:
    Optimizer(float learning_rate, const std::vector<std::shared_ptr<Tensor>>& parameters) : _learning_rate(learning_rate), _parameters(parameters) {}
    virtual void step() = 0;
    void zero_grad();
    void add_parameter(std::shared_ptr<Tensor> param) { _parameters.push_back(param); }
    const std::vector<std::shared_ptr<Tensor>>& parameters() const { return _parameters; }
    float learning_rate() const { return _learning_rate; }
};