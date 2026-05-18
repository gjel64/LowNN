#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include "core/Math/vecUtils.hpp"

/*
must TODO:
    - broadcasting
    - operations
    - indexing

wanted TODO:
    - print with shape
    - negative indexing 
    - slicing (struct Slice { int start, int end, int step } and operator[](Slice...))
*/


class Tensor {
private:

    // -------- Attributes --------
    std::vector<float> _data;
    std::vector<std::size_t> _shape;
    std::vector<std::size_t> _strides;
    std::size_t _size;

    // -------- Private Methods --------
    

public:
    // -------- Init --------
    template<typename V>
    Tensor(V data){
        if constexpr (std::is_floating_point_v<V>) 
        {
            _data = std::vector{data};
            _size = _data.size();
            _shape = {_data.size()};
            _strides = {};
        } 
        else if constexpr (is_std_vector<V>::value) 
        {
            VecUtils::flatten_into(data, _data);
            _size = _data.size();
            _shape = VecUtils::get_shape(data);
            _strides = std::vector<std::size_t>{1};
            for (std::size_t i = 0; i < _shape.size() - 1; i++) {
                _strides.insert(_strides.begin(), _shape[i + 1] * _strides.front());
            }
            
        }
        else 
        {
            throw std::invalid_argument("Fail at init : Unsupported data type for Tensor");
        }

        // Test if data shape is good
        std::size_t tot_shape_size = 1;
        for (std::size_t s : _shape){
            tot_shape_size *= s;
        }
        if (_size != tot_shape_size) {
            throw std::invalid_argument("Fail at init : Data need a regular shape");
        }
    }
    Tensor(std::vector<float> data, std::vector<std::size_t> shape) : _data(data), _shape(shape) {
        _size = _data.size();
        _strides = std::vector<std::size_t>{1};
        for (std::size_t i = 0; i < _shape.size() - 1; i++) {
            _strides.insert(_strides.begin(), _shape[i + 1] * _strides.front());
        }
    }

    // -------- Methods --------
    const float item() const;
    const std::string to_string() const;
    void print();

    // -------- Indexing --------
    template <typename... Args>
    std::shared_ptr<Tensor> operator[](Args... indices) {
        if (sizeof...(indices) > _shape.size()) {
            throw std::invalid_argument("Too many indices for tensor");
        }


        // only support int indexing for now
        std::size_t index_of_data = 0;
        std::vector<int> vec = {indices...};
        for (std::size_t i = 0; i < vec.size(); i++)
        {
            if (vec[i] < 0 || vec[i] >= _shape[i]) {
                throw std::out_of_range("Index out of range");
            }
            index_of_data += _strides[i] * vec[i];
        }

        return std::make_shared<Tensor>(_data[index_of_data]);
    }

    // -------- Getters --------
    const std::vector<float>& data() const { return _data; }
    const std::vector<std::size_t>& shape() const { return _shape; }
    const std::vector<std::size_t>& strides() const { return _strides; }
    std::size_t size() const { return _size; }

};