#include "core/tensor.hpp"

const float Tensor::item() const 
{ 
    if (_size != 1) {
        throw std::runtime_error("Tensor does not contain exactly one item");
    }
    return _data[0]; 
}

const std::string Tensor::to_string() const 
{
    std::string str = "Tensor ( ";

    // data
    str += "data: ";
    str += "[";
    for (std::size_t i = 0; i < _size; i++) {
        str += std::to_string(_data[i]);
        if (i < _size - 1) {
            str += ", ";
        }
    }
    str += "] ";


    str += ")";
    return str;
}

void Tensor::print() 
{
    std::cout << to_string() << "\n";
}
