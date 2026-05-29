#pragma once
#include <memory>
#include <iostream>
#include <cmath>
#include "core/grad_fn.hpp"
#include "core/Math/vecUtils.hpp"
#include "core/GradFn/add_grad_fn.hpp"
#include "core/engine.hpp"

/* must TODO:
    - indexing with : and negative indexing
    - autograd (grad_fn, grad_accumulator, output_nr)
    - flatten method (remove 1s dims)
    - add "with no_grad"
    - create tensor full of a value with shape

wanted TODO:
    - internal iterator over the real data (to avoid strides / offset code duplication)
*/

class Tensor: public std::enable_shared_from_this<Tensor> {
private:

    // -------- Attributes --------
    std::shared_ptr<float[]> _pdata;
    std::vector<std::size_t> _shape;
    std::vector<std::size_t> _strides;
    std::size_t _size;
    std::size_t _offset = 0;
    std::shared_ptr<float[]> _pgrad;
    bool _require_grad;
    std::shared_ptr<GradFn> _grad_fn;
    std::vector<std::shared_ptr<Tensor>> _parents;

    // -------- Private methods --------
    void _set_grad_fn(std::shared_ptr<GradFn> grad_fn) { _grad_fn = grad_fn; }



public:
    // -------- Init --------
    template<typename V>
    Tensor(V data, bool require_grad = false) : _offset(0), _require_grad(require_grad){
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
        if (_require_grad) {
            _pgrad = std::make_shared<float[]>(_size);
        }
    }
    Tensor(std::shared_ptr<float[]> data, std::vector<std::size_t> shape, std::size_t offset, std::vector<std::size_t> strides, bool require_grad = false, std::vector<std::shared_ptr<Tensor>> parents = {});
    Tensor(std::shared_ptr<float[]> data, std::vector<std::size_t> shape, std::size_t offset, std::size_t size, bool require_grad = false, std::vector<std::shared_ptr<Tensor>> parents = {}); 

    // -------- Methods --------
    const float item() const;
    const std::string to_string() const;
    void print();   
    std::shared_ptr<Tensor> squeeze();
    void backward(std::shared_ptr<Tensor> grad = nullptr);
    void ensure_grad_allocated();

    // ------- static Methods --------
    template <typename... Args>
    static Tensor empty(Args... indices) {
        std::vector<int> vec = {indices...};
        std::vector<std::size_t> shape(vec.begin(), vec.end());
        std::size_t size = 1;
        if (shape.empty()) {
            size = 0;
        } 
        else {
            for (std::size_t s : shape){
                size *= s;
            }
        }
        std::shared_ptr<float[]> data = std::make_shared<float[]>(size);
        return Tensor(data, shape, 0, size);
    } 
    static Tensor fill(float value, const std::vector<std::size_t>& shape);
    

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
    std::shared_ptr<Tensor> pow(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> sqrt();
    std::shared_ptr<Tensor> operator/ (std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> other);

    // -------- Getters --------
    const std::shared_ptr<float[]> data() const { return _pdata; }
    const std::vector<std::size_t>& shape() const { return _shape; }
    const std::vector<std::size_t>& strides() const { return _strides; }
    const std::size_t size() const { return _size; }
    const std::size_t offset() const { return _offset; }
    const bool require_grad() const { return _require_grad; }
    std::vector<std::shared_ptr<Tensor>> parents() { return _parents; }
    std::shared_ptr<float[]> gradp() { return _pgrad; }
    std::shared_ptr<GradFn> grad_fn() { return _grad_fn; }

    // -------- Setters --------
    void set_require_grad(bool require_grad) { _require_grad = require_grad; if (_require_grad && !_pgrad) { _pgrad = std::make_shared<float[]>(_size); } }

};