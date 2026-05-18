#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <typeinfo>
#include <type_traits>
#include "core/Math/vecUtils.hpp"

/*
TODO:
    - print with shape
    - negative indexing
    - broadcasting
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
            throw std::invalid_argument("Unsupported data type for Tensor");
        }
    }

    // -------- Methods --------
    const float item() const;
    const std::string to_string() const;
    void print();


    // -------- Getters --------
    const std::vector<float>& data() const { return _data; }
    const std::vector<std::size_t>& shape() const { return _shape; }
    const std::vector<std::size_t>& strides() const { return _strides; }
    std::size_t size() const { return _size; }
};