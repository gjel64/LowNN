#include "core/grad_fn.hpp"

GradFn::GradFn(std::string name) : _name(name) {
    if (name == "add") {
        _fn = [](Tensor& t) {
            
        };
    }
    else if (name == "mul") {
        _fn = [](Tensor& t) {
            
        };
    }
    else if (name == "sub") {
        _fn = [](Tensor& t) {
            
        };
    }
    else if (name == "pow") {
        _fn = [](Tensor& t) {
            
        };
    }
    else if (name == "sqrt") {
        _fn = [](Tensor& t) {
            
        };
    }
    else if (name == "div") {
        _fn = [](Tensor& t) {
            
        };
    }
    else if (name == "matmul") {
        _fn = [](Tensor& t) {
            
        };
    }
    else if (name == "slice") {
        _fn = [](Tensor& t) {
            
        };
    }
    else {
        throw std::invalid_argument("Unknown grad_fn name");
    }
}