#include "core/Utils/vecUtils.hpp"

std::vector<size_t> VecUtils::broadcast(const std::vector<size_t>& shape1_in, std::vector<size_t> shape2){
    std::vector<size_t> shape1 = shape1_in;
    
    // fill smaller shape with 1
    while (shape1.size() < shape2.size()) {
        shape1.insert(shape1.begin(), 1); 
    }
    while (shape2.size() < shape1.size()) {
        shape2.insert(shape2.begin(), 1); 
    }

    std::vector<size_t> result_shape(shape1.size());
    for (size_t i = 0; i < shape1.size(); ++i) {
        if (shape1[i] == shape2[i]) {
            result_shape[i] = shape1[i];
        } 
        else if (shape1[i] == 1) {
            result_shape[i] = shape2[i];
        } 
        else if (shape2[i] == 1) {
            result_shape[i] = shape1[i];
        } 
        else {
            throw std::invalid_argument("Shapes are not compatible for broadcasting");
        }
    }
    return result_shape;
}

std::vector<size_t> VecUtils::squeeze_shape(std::vector<size_t> shape) {
    std::vector<size_t> squeezed_shape;
    for (size_t dim : shape) {
        if (dim != 1) {
            squeezed_shape.push_back(dim);
        }
    }
    return squeezed_shape;
}