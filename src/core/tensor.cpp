#include "core/tensor.hpp"



// ------------------------ Init ------------------------

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

