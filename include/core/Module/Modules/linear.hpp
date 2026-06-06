#pragma once 
#include "core/Module/module.hpp"

class Linear : public Module {
private:
    std::shared_ptr<Tensor> _weight;
    std::shared_ptr<Tensor> _bias;
public:
    Linear(std::size_t in_features, std::size_t out_features, bool bias=true, float gain=1.0f);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;

    void set_weight(std::shared_ptr<Tensor> new_weight);
    void set_bias(std::shared_ptr<Tensor> new_bias);
};