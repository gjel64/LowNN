#include "core/Module/Modules/linear.hpp"

Linear::Linear(std::size_t in_features, std::size_t out_features, bool bias, float gain) {
    // init weights with Kaiming He initialization
    weight = std::make_shared<Tensor>(Tensor::randn(
        std::vector<std::size_t>{in_features, out_features}, 0.0f,  (1.0f / std::sqrt(in_features) * gain)
    ));

    if (bias) {
        this->bias = std::make_shared<Tensor>(Tensor::randn(std::vector<std::size_t>{out_features}));
    }
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input) {
    std::shared_ptr<Tensor> output = input->matmul(weight);

    if (bias) {
        output = (*output) + bias ;
    }

    return output;
}

std::vector<std::shared_ptr<Tensor>> Linear::parameters() {
    if (bias) {
        return std::vector<std::shared_ptr<Tensor>>{weight, bias};
    }
    else {
        return std::vector<std::shared_ptr<Tensor>>{weight};
    }
}