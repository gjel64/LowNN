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
    return "to_string not implemented yet";
}

void Tensor::print() 
{
    std::cout << to_string() << "\n";
}



// ------------------------ Operators ------------------------ 

std::shared_ptr<Tensor> Tensor::operator+ (std::shared_ptr<Tensor> other){
        std::vector<std::size_t> result_shape = VecUtils::broadcast(this->shape(), other->shape());    

        
}