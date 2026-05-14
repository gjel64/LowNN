#include "core/tensor.h"



// -------------------- Init --------------------

Tensor::Tensor(float data) {
    _data.push_back(data);
    _shape.push_back(1);
    _strides.push_back(1);
    _size = 1;
}

Tensor::Tensor(std::vector<float> data) {
    _data = data;
    _shape.push_back(data.size());
    _strides.push_back(1);
    _size = data.size();
}

Tensor::Tensor(std::vector<std::vector<float>> data) {
    _shape.push_back(data.size());
    _shape.push_back(data[0].size());
    _strides.push_back(_shape[1]);
    _strides.push_back(1);
    _size = _shape[0] * _shape[1];
    for (std::vector<float> row : data) {
        _data.insert(_data.end(), row.begin(), row.end());
    }
}



// -------------------- Getters / Setters --------------------

float Tensor::item() const {
    if (_size != 1) {
        throw std::invalid_argument("Tensor has more than one element");
    }
    return _data[0];
}

std::shared_ptr<Tensor> Tensor::operator[](int index) {
    if (index < 0 || index > _shape[0]) {
        throw std::out_of_range("Index out of range");
    }
    if (_shape.size() == 1) {
        return std::make_shared<Tensor>(_data[index]);
    }
    if (_shape.size() == 2) {
        std::vector<float> row(_data.begin() + index * _strides[0], _data.begin() + (index + 1) * _strides[0]);
        return std::make_shared<Tensor>(row);
    }
    throw std::invalid_argument("Unsupported tensor dimension");
}

std::vector<std::size_t> Tensor::shape() const {
    return _shape;
}

std::vector<std::size_t> Tensor::strides() const {
    return _strides;
}

std::size_t Tensor::size() const {
    return _size;
}



