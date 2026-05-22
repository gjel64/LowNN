#pragma once
#include <memory>
#include <iostream>
#include <cmath>
#include "core/Math/vecUtils.hpp"

/*
must TODO:
    - operations (-,*,/, matmul, etc.)
    - indexing with : / negative indexing
    - autograd (grad_fn, grad_accumulator, output_nr)
    - flatten method (remove 1 dims)

wanted TODO:
    - print with shape
*/


class Tensor {
private:

    // -------- Attributes --------
    std::shared_ptr<float[]> _pdata;
    std::vector<std::size_t> _shape;
    std::vector<std::size_t> _strides;
    std::size_t _size;
    std::size_t _offset = 0; // slicing offset
    

public:
    // -------- Init --------
    template<typename V>
    Tensor(V data) : _offset(0){
        if constexpr (std::is_floating_point_v<V>) 
        {
            _pdata = std::make_shared<float[]>(1);
            _pdata[0] = data;
            _size = 1;
            _shape = {1};
            _strides = {};
        } 
        else if constexpr (is_std_vector<V>::value) 
        {
            std::vector<float> flat_data;
            VecUtils::flatten_into(data, flat_data);

            _pdata = std::make_shared<float[]>(flat_data.size());
            std::copy(flat_data.begin(), flat_data.end(), _pdata.get());
            
            _size = flat_data.size();
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
    Tensor(std::shared_ptr<float[]> data, std::vector<std::size_t> shape, std::size_t offset, std::vector<std::size_t> strides); 

    // -------- Methods --------
    const float item() const;
    const std::string to_string() const;
    void print();

    // -------- Indexing --------
    template <typename... Args> // auto-depth indexing with offset / stride / variadic template
    std::shared_ptr<Tensor> operator[](Args... indices) {
        std::vector<int> vec = {indices...};

        if (vec.size() > _shape.size() || vec.size() == 0){
            throw std::invalid_argument("Too many indices for tensor");
        }
        else if (sizeof...(indices) == _shape.size()) 
        {
            // recup index of data
            std::size_t index_of_data = 0;
            for (std::size_t i = 0; i < vec.size(); i++)
            {
                if (vec[i] < 0 || vec[i] >= _shape[i]) {
                    throw std::out_of_range("Index out of range");
                }
                index_of_data += _strides[i] * vec[i];
            }
            return std::make_shared<Tensor>(_pdata, std::vector<std::size_t>{1}, _offset + index_of_data, std::vector<std::size_t>{});
        }
        else //(sizeof...(indices) < _shape.size()) -> need a vector 
        {

            std::size_t offset = 0;
            std::size_t i = 0;
            for (auto indice : vec) 
            {
                offset += _strides[i] * indice;
                i ++;
            }

            std::vector<std::size_t> shape(_shape.begin() + vec.size(), _shape.end());
            std::vector<std::size_t> strides(_strides.begin() + vec.size(), _strides.end());

            return std::make_shared<Tensor>(_pdata, shape, offset, strides);
        }
    }

    // -------- Operators --------
    std::shared_ptr<Tensor> operator+ (std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> operator* (std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> operator- (std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> pow (std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> operator/ (std::shared_ptr<Tensor> other);



    // -------- Getters --------
    const std::shared_ptr<float[]> data() const { return _pdata; }
    const std::vector<std::size_t>& shape() const { return _shape; }
    const std::vector<std::size_t>& strides() const { return _strides; }
    std::size_t size() const { return _size; }
    std::size_t offset() const { return _offset; }

};