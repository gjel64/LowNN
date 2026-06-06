#pragma once
#include <vector>
#include <memory>
#include "core/tensor.hpp"
#include "core/optimizer.hpp"


class SGD : public Optimizer {
private:
    bool _maximize;
public:
    SGD(float learning_rate, 
        const std::vector<std::shared_ptr<Tensor>>& parameters,
        bool maximize = false
    );
    void step() override;
};

