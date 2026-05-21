#include "core/Math/vecUtils.hpp"

std::vector<size_t> VecUtils::broadcast(const std::vector<size_t>& shape1, std::vector<size_t> shape2){
    if (shape1.size() < shape2.size() || shape2.size() == 0) {
        throw std::invalid_argument("Shapes must have the same number of dimensions for broadcasting and shape1 must be the larger one");
    }
    while (shape2.size() < shape1.size()) {
        shape2.insert(shape2.begin(), 1); 
    }

    std::vector<size_t> result_shape(shape1.size());
    for (size_t i = 0; i < shape1.size(); ++i) {
        if (shape2.size() < i + 1) {
            result_shape[i] = shape1[i]; 
        } 
        else if (shape1[i] == shape2[i]) {
            result_shape[i] = shape1[i];
        } 
        else if (shape2[i] == 1 ) {
            result_shape[i] = shape1[i];
        } 
        else {
            throw std::invalid_argument("Shapes are not compatible for broadcasting");
        }
    }
    return result_shape;
}