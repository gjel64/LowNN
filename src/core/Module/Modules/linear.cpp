#include "core/Module/Modules/linear.hpp"

Linear::Linear(std::size_t in_features, std::size_t out_features, bool bias, float gain) {
    // init weights with Kaiming He initialization
    _weight = std::make_shared<Tensor>(Tensor::randn(
        std::vector<std::size_t>{in_features, out_features}, 0.0f,  (1.0f / std::sqrt(in_features) * gain)
    ));

    _weight->set_require_grad(true);

    if (bias) {
        _bias = std::make_shared<Tensor>(Tensor::randn(std::vector<std::size_t>{out_features}));
        _bias->set_require_grad(true);
    }
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input) {
    std::shared_ptr<Tensor> output = input->matmul(_weight);

    if (_bias) {
        output = (*output) + _bias ;
    }

    return output;
}

std::vector<std::shared_ptr<Tensor>> Linear::parameters() {
    if (_bias) {
        return std::vector<std::shared_ptr<Tensor>>{_weight, _bias};
    }
    else {
        return std::vector<std::shared_ptr<Tensor>>{_weight};
    }
}

void Linear::set_weight(std::shared_ptr<Tensor> new_weight) {
    _weight = new_weight;
    if (!_weight->require_grad()) {
        _weight->set_require_grad(true);
    }
}

void Linear::set_bias(std::shared_ptr<Tensor> new_bias) {
    _bias = new_bias;
    if (!_bias->require_grad()) {
        _bias->set_require_grad(true);
    }
}