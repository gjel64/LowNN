#include "core/tensor.hpp"



// ------------------------ Init ------------------------

Tensor::Tensor(std::shared_ptr<float[]> data, std::vector<std::size_t> shape, 
    std::size_t offset, std::vector<std::size_t> strides, 
    bool require_grad, std::vector<std::shared_ptr<Tensor>> parents) 
    : _pdata(data), _shape(shape), _offset(offset), _strides(strides), _require_grad(require_grad), _grad_fn(nullptr), _grad(nullptr), _parents(parents)
{
    std::size_t size = 1;
    for (std::size_t s : shape){
        size *= s;
    }
    _size = size;

    if (_require_grad) {
        _grad = std::make_shared<float[]>(size);
    }
}

Tensor::Tensor(std::shared_ptr<float[]> data, std::vector<std::size_t> shape, 
    std::size_t offset, std::size_t size, bool require_grad, 
    std::vector<std::shared_ptr<Tensor>> parents) 
    : _pdata(data), _shape(shape), _offset(offset), _size(size), _require_grad(require_grad), _grad_fn(nullptr), _grad(nullptr), _parents(parents)
{
    // calc strides
    _strides = std::vector<std::size_t>(shape.size(), 1);
    if (shape.size() > 0) {
        for (int i = shape.size() - 2; i >= 0; i--) {
            _strides[i] = shape[i + 1] * _strides[i + 1];
        }
    }
    if (_require_grad) {
        _grad = std::make_shared<float[]>(size);
    }

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

std::shared_ptr<Tensor> Tensor::squeeze() 
{
    // remove all 1s dims
    std::vector<std::size_t> flat_shape;
    for (std::size_t dim : _shape) {
        if (dim != 1) {
            flat_shape.push_back(dim);
        }
    }
    return std::make_shared<Tensor>(_pdata, flat_shape, _offset, _size);
}

void Tensor::backward() 
{
    if (!_require_grad) {
        throw std::runtime_error("Cannot call backward on a tensor that does not require grad");
    }
    if (_grad_fn) {
       _grad_fn->operator()(*this);
    }
}


// ------------------------ Operators ------------------------ 

std::shared_ptr<Tensor> Tensor::operator+ (std::shared_ptr<Tensor> other){
    
    const std::shared_ptr<float[]> other_data = other->data();
    const std::size_t other_offset = other->offset();
    const std::vector<std::size_t> other_strides = other->strides();
    const std::vector<std::size_t> other_shape = other->shape();

    std::vector<std::size_t> result_shape = VecUtils::broadcast(this->shape(), other->shape());

    // calc strides of result
    std::vector<std::size_t> result_stride(result_shape.size(), 1);
    if (result_shape.size() > 0) {
        for (int i = result_shape.size() - 2; i >= 0; i--) {
            result_stride[i] = result_shape[i + 1] * result_stride[i + 1];
        }
    }

    // calc size of result
    std::size_t result_size = 1;
    for (std::size_t dim_size : result_shape){
        result_size *= dim_size;
    }

    std::shared_ptr<float[]> result_data = std::make_shared<float[]>(result_size);

    if (result_size > 0) {
        // calc strides for this and other with broadcasting
        std::vector<std::size_t> this_b_strides(result_shape.size(), 0);
        std::vector<std::size_t> other_b_strides(result_shape.size(), 0);

        int this_dim_diff = result_shape.size() - _shape.size();
        int other_dim_diff = result_shape.size() - other_shape.size();

        for (std::size_t dim = 0; dim < result_shape.size(); dim++) {
            if (dim >= this_dim_diff) {
                std::size_t this_dim = dim - this_dim_diff;
                if (_shape[this_dim] != 1) {
                    this_b_strides[dim] = _strides[this_dim];
                }
            }

            if (dim >= other_dim_diff) {
                std::size_t other_dim = dim - other_dim_diff;
                if (other_shape[other_dim] != 1) { 
                    other_b_strides[dim] = other_strides[other_dim];
                }
            }
        }

        std::vector<std::size_t> indices(result_shape.size(), 0);
        std::size_t this_i = 0;
        std::size_t other_i = 0;

        // main loop to fill result data
        for (std::size_t i = 0; i < result_size; i++) {
            result_data[i] = _pdata[this_i + _offset] + other_data[other_i + other_offset];

            for (int dim = result_shape.size() - 1; dim >= 0; dim--) {
                indices[dim]++;
                if (indices[dim] < result_shape[dim]) {
                    this_i += this_b_strides[dim];
                    other_i += other_b_strides[dim];
                    break;
                } 
                else {
                    indices[dim] = 0;
                    this_i -= this_b_strides[dim] * (result_shape[dim] - 1);
                    other_i -= other_b_strides[dim] * (result_shape[dim] - 1);
                }
            }
        }
    }

    return std::make_shared<Tensor>(result_data, result_shape, 0, result_stride);    
}

std::shared_ptr<Tensor> Tensor::operator* (std::shared_ptr<Tensor> other){
    
    const std::shared_ptr<float[]> other_data = other->data();
    const std::size_t other_offset = other->offset();
    const std::vector<std::size_t> other_strides = other->strides();
    const std::vector<std::size_t> other_shape = other->shape();

    std::vector<std::size_t> result_shape = VecUtils::broadcast(this->shape(), other->shape());

    // calc strides of result
    std::vector<std::size_t> result_stride(result_shape.size(), 1);
    if (result_shape.size() > 0) {
        for (int i = result_shape.size() - 2; i >= 0; i--) {
            result_stride[i] = result_shape[i + 1] * result_stride[i + 1];
        }
    }

    // calc size of result
    std::size_t result_size = 1;
    for (std::size_t dim_size : result_shape){
        result_size *= dim_size;
    }

    std::shared_ptr<float[]> result_data = std::make_shared<float[]>(result_size);

    if (result_size > 0) {
        // calc strides for this and other with broadcasting
        std::vector<std::size_t> this_b_strides(result_shape.size(), 0);
        std::vector<std::size_t> other_b_strides(result_shape.size(), 0);

        int this_dim_diff = result_shape.size() - _shape.size();
        int other_dim_diff = result_shape.size() - other_shape.size();

        for (std::size_t dim = 0; dim < result_shape.size(); dim++) {
            if (dim >= this_dim_diff) {
                std::size_t this_dim = dim - this_dim_diff;
                if (_shape[this_dim] != 1) {
                    this_b_strides[dim] = _strides[this_dim];
                }
            }

            if (dim >= other_dim_diff) {
                std::size_t other_dim = dim - other_dim_diff;
                if (other_shape[other_dim] != 1) { 
                    other_b_strides[dim] = other_strides[other_dim];
                }
            }
        }

        std::vector<std::size_t> indices(result_shape.size(), 0);
        std::size_t this_i = 0;
        std::size_t other_i = 0;

        // main loop to fill result data
        for (std::size_t i = 0; i < result_size; i++) {
            result_data[i] = _pdata[this_i + _offset] * other_data[other_i + other_offset];

            for (int dim = result_shape.size() - 1; dim >= 0; dim--) {
                indices[dim]++;
                if (indices[dim] < result_shape[dim]) {
                    this_i += this_b_strides[dim];
                    other_i += other_b_strides[dim];
                    break;
                } 
                else {
                    indices[dim] = 0;
                    this_i -= this_b_strides[dim] * (result_shape[dim] - 1);
                    other_i -= other_b_strides[dim] * (result_shape[dim] - 1);
                }
            }
        }
    }

    return std::make_shared<Tensor>(result_data, result_shape, 0, result_stride);    
}

std::shared_ptr<Tensor> Tensor::operator- (std::shared_ptr<Tensor> other){
    return (*this) + ((*other) * std::make_shared<Tensor>(-1.0f));
}

std::shared_ptr<Tensor> Tensor::pow (std::shared_ptr<Tensor> other){
    
    const std::shared_ptr<float[]> other_data = other->data();
    const std::size_t other_offset = other->offset();
    const std::vector<std::size_t> other_strides = other->strides();
    const std::vector<std::size_t> other_shape = other->shape();

    std::vector<std::size_t> result_shape = VecUtils::broadcast(this->shape(), other->shape());

    // calc strides of result
    std::vector<std::size_t> result_stride(result_shape.size(), 1);
    if (result_shape.size() > 0) {
        for (int i = result_shape.size() - 2; i >= 0; i--) {
            result_stride[i] = result_shape[i + 1] * result_stride[i + 1];
        }
    }

    // calc size of result
    std::size_t result_size = 1;
    for (std::size_t dim_size : result_shape){
        result_size *= dim_size;
    }

    std::shared_ptr<float[]> result_data = std::make_shared<float[]>(result_size);

    if (result_size > 0) {
        // calc strides for this and other with broadcasting
        std::vector<std::size_t> this_b_strides(result_shape.size(), 0);
        std::vector<std::size_t> other_b_strides(result_shape.size(), 0);

        int this_dim_diff = result_shape.size() - _shape.size();
        int other_dim_diff = result_shape.size() - other_shape.size();

        for (std::size_t dim = 0; dim < result_shape.size(); dim++) {
            if (dim >= this_dim_diff) {
                std::size_t this_dim = dim - this_dim_diff;
                if (_shape[this_dim] != 1) {
                    this_b_strides[dim] = _strides[this_dim];
                }
            }

            if (dim >= other_dim_diff) {
                std::size_t other_dim = dim - other_dim_diff;
                if (other_shape[other_dim] != 1) { 
                    other_b_strides[dim] = other_strides[other_dim];
                }
            }
        }

        std::vector<std::size_t> indices(result_shape.size(), 0);
        std::size_t this_i = 0;
        std::size_t other_i = 0;

        // main loop to fill result data
        for (std::size_t i = 0; i < result_size; i++) {
            result_data[i] = std::pow(_pdata[this_i + _offset], other_data[other_i + other_offset]);

            for (int dim = result_shape.size() - 1; dim >= 0; dim--) {
                indices[dim]++;
                if (indices[dim] < result_shape[dim]) {
                    this_i += this_b_strides[dim];
                    other_i += other_b_strides[dim];
                    break;
                } 
                else {
                    indices[dim] = 0;
                    this_i -= this_b_strides[dim] * (result_shape[dim] - 1);
                    other_i -= other_b_strides[dim] * (result_shape[dim] - 1);
                }
            }
        }
    }

    return std::make_shared<Tensor>(result_data, result_shape, 0, result_stride);    
}

std::shared_ptr<Tensor> Tensor::operator/ (std::shared_ptr<Tensor> other){
    return (*this) * other->pow(std::make_shared<Tensor>(-1.0f));  
}

std::shared_ptr<Tensor> Tensor::sqrt()
{
    return this->pow(std::make_shared<Tensor>(0.5f));
}

std::shared_ptr<Tensor> Tensor::matmul(std::shared_ptr<Tensor> other)
{
    const std::vector<std::size_t>& other_shape = other->shape();
    const std::vector<std::size_t>& other_strides = other->strides();
    const std::shared_ptr<float[]> other_data = other->data();
    const std::size_t other_offset = other->offset();

    // (m?, n), (n,p?) -> (m?, p?)
    std::size_t this_n = _shape.back();
    std::size_t other_n = other_shape.at(0);
    std::size_t other_s_size = other_shape.size();
    if (other_s_size >= 2) {
        other_n = other_shape.at(other_s_size-2);
    }

    if (this_n != other_n) {
        throw std::invalid_argument("(m?, n), (n,p?) -> (m?, p?) : n needs to be the same");
    }
    std::size_t n = this_n;

    // (m?, n), (n,p?) -> (m?, p?) (broadcast m? and p?)
    std::vector<std::size_t> m = {};
    std::vector<std::size_t> p = {};
    if (_shape.size() > 2) {
        m = {_shape.begin(), _shape.end() - 2};
    }
    if (other_s_size > 2) {
        p = {other_shape.begin(), other_shape.end() - 2};
    }

    std::vector<std::size_t> batch_shape = VecUtils::broadcast(m, p);
    std::vector<std::size_t> result_shape = batch_shape;
    if (_shape.size() >= 2) {
        result_shape.push_back(_shape[_shape.size() - 2]);
    }
    if (other_shape.size() >= 2) {
        result_shape.push_back(other_shape[other_shape.size() - 1]);
    }

    // calc size of result
    std::size_t result_size = 1;
    for (std::size_t dim_size : result_shape){
        result_size *= dim_size;
    }

    std::shared_ptr<float[]> result_data = std::make_shared<float[]>(result_size);

    if (result_size > 0) {
        const std::size_t this_dim = _shape.size();
        const std::size_t other_dim = other_shape.size();

        const std::size_t a_k_stride = _strides[this_dim - 1];
        const std::size_t b_k_stride = other_dim >= 2 ? other_strides[other_dim - 2] : other_strides[0];
        const std::size_t a_row_stride = this_dim >= 2 ? _strides[this_dim - 2] : 0;
        const std::size_t b_col_stride = other_dim >= 2 ? other_strides[other_dim - 1] : 0;

        std::vector<std::size_t> this_batch_strides(batch_shape.size(), 0);
        std::vector<std::size_t> other_batch_strides(batch_shape.size(), 0);

        std::size_t this_batch_diff = batch_shape.size() - m.size();
        std::size_t other_batch_diff = batch_shape.size() - p.size();

        for (std::size_t dim = 0; dim < batch_shape.size(); dim++) {
            if (dim >= this_batch_diff) {
                std::size_t this_batch_dim = dim - this_batch_diff;
                if (m[this_batch_dim] != 1) {
                    this_batch_strides[dim] = _strides[this_batch_dim];
                }
            }
            if (dim >= other_batch_diff) {
                std::size_t other_batch_dim = dim - other_batch_diff;
                if (p[other_batch_dim] != 1) {
                    other_batch_strides[dim] = other_strides[other_batch_dim];
                }
            }
        }

        std::vector<std::size_t> result_indices(result_shape.size(), 0);
        const std::size_t batch_ndim = batch_shape.size();

        for (std::size_t out_i = 0; out_i < result_size; out_i++) {
            std::size_t this_base = _offset;
            std::size_t other_base = other_offset;

            for (std::size_t dim = 0; dim < batch_ndim; dim++) {
                this_base += result_indices[dim] * this_batch_strides[dim];
                other_base += result_indices[dim] * other_batch_strides[dim];
            }

            if (this_dim >= 2) {
                this_base += result_indices[batch_ndim] * a_row_stride;
            }
            if (other_dim >= 2) {
                std::size_t col_index_pos = batch_ndim + (this_dim >= 2 ? 1 : 0);
                other_base += result_indices[col_index_pos] * b_col_stride;
            }

            float sum = 0.0f;
            for (std::size_t k = 0; k < n; k++) {
                sum += _pdata[this_base + k * a_k_stride] * other_data[other_base + k * b_k_stride];
            }
            result_data[out_i] = sum;

            if (!result_shape.empty()) {
                for (int dim = static_cast<int>(result_shape.size()) - 1; dim >= 0; dim--) {
                    result_indices[dim]++;
                    if (result_indices[dim] < result_shape[dim]) {
                        break;
                    }
                    result_indices[dim] = 0;
                }
            }
        }
    }


    return std::make_shared<Tensor>(result_data, result_shape, 0, result_size);
}



