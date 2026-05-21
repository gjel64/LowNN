#include "core/tensor.hpp"


Tensor::Tensor(std::shared_ptr<float[]> data, std::vector<std::size_t> shape, std::size_t offset, std::vector<std::size_t> strides) 
    : _pdata(data), _shape(shape), _offset(offset), _strides(strides) 
{
    std::size_t size = 1;
    for (std::size_t s : shape){
        size *= s;
    }
    _size = size;
}


// ------------------------ Methods ------------------------

const float Tensor::item() const 
{ 
    if (_size != 1) {
        throw std::runtime_error("Tensor does not contain one item");
    }
    return _pdata[0 + _offset]; 
}

const std::string Tensor::to_string() const 
{
    std::string str = "Tensor ( ";

    // data
    str += "data: ";
    str += "[";
    for (std::size_t i = 0; i < _size; i++) {
        str += std::to_string(_pdata[i]);
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



// ------------------------ Private Methods ------------------------ 