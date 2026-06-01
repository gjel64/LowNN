#include "core/GradFn/matmul_grad_fn.hpp"
#include "core/tensor.hpp"

MatMulGradFn::MatMulGradFn(std::vector<std::weak_ptr<Tensor>> inputs, std::weak_ptr<Tensor> output)
    : GradFn("matmul", inputs, output)
{
}

void MatMulGradFn::backward(
    const std::shared_ptr<Tensor>& out,
    const std::shared_ptr<Tensor>& out_grad
){
    auto a = inputs[0].lock();
    auto b = inputs[1].lock();
    if (!a || !b) {
        throw std::runtime_error("MatMulGradFn: input tensors no longer exist");
    }
    a->ensure_grad_allocated();
    b->ensure_grad_allocated();
    std::shared_ptr<float[]> a_grad_data = a->gradp();
    std::shared_ptr<float[]> b_grad_data = b->gradp();
    
    std::size_t size_max = a->size();
    if (b->size() > size_max) {
        size_max = b->size();
    }
    // 1D @ 1D
    if (VecUtils::squeeze_shape(b->shape()).size() == 1 && VecUtils::squeeze_shape(a->shape()).size() == 1) {
        for (std::size_t i = 0; i < size_max; i++) {
            a_grad_data.get()[i % a->size()] += out_grad->data().get()[(i + out_grad->offset()) % out_grad->size()] * b->data().get()[i % b->size()];
            b_grad_data.get()[i % b->size()] += out_grad->data().get()[(i + out_grad->offset()) % out_grad->size()] * a->data().get()[i % a->size()];
        }
    }
    else {
        // (..., M, N) @ (..., N, P) -> (..., M, P)
        const std::vector<std::size_t>& a_shape = a->shape();
        const std::vector<std::size_t>& b_shape = b->shape();
        const std::vector<std::size_t>& out_shape = out->shape();

        // ensure grads allocated
        a->ensure_grad_allocated();
        b->ensure_grad_allocated();
        
        std::size_t this_dim = a_shape.size();
        std::size_t other_dim = b_shape.size();

        std::size_t a_k_stride = this_dim >= 1 ? a->strides()[this_dim - 1] : 1;
        std::size_t b_k_stride = other_dim >= 2 ? b->strides()[other_dim - 2] : b->strides()[0];
        std::size_t a_row_stride = this_dim >= 2 ? a->strides()[this_dim - 2] : 0;
        std::size_t b_col_stride = other_dim >= 2 ? b->strides()[other_dim - 1] : 0;

        // batch shapes
        std::vector<std::size_t> m = {};
        std::vector<std::size_t> p = {};
        if (a_shape.size() > 2) {
            m = {a_shape.begin(), a_shape.end() - 2};
        }
        if (b_shape.size() > 2) {
            p = {b_shape.begin(), b_shape.end() - 2};
        }

        std::vector<std::size_t> batch_shape = VecUtils::broadcast(m, p);

        std::vector<std::size_t> this_batch_strides(batch_shape.size(), 0);
        std::vector<std::size_t> other_batch_strides(batch_shape.size(), 0);

        std::size_t this_batch_diff = batch_shape.size() - m.size();
        std::size_t other_batch_diff = batch_shape.size() - p.size();

        for (std::size_t dim = 0; dim < batch_shape.size(); dim++) {
            if (dim >= this_batch_diff) {
                std::size_t this_batch_dim = dim - this_batch_diff;
                if (m[this_batch_dim] != 1) {
                    this_batch_strides[dim] = a->strides()[this_batch_dim];
                }
            }
            if (dim >= other_batch_diff) {
                std::size_t other_batch_dim = dim - other_batch_diff;
                if (p[other_batch_dim] != 1) {
                    other_batch_strides[dim] = b->strides()[other_batch_dim];
                }
            }
        }

        std::vector<std::size_t> result_indices(out_shape.size(), 0);
        const std::size_t batch_ndim = batch_shape.size();
        std::size_t n = a_shape.back();

        for (std::size_t out_i = 0; out_i < out->size(); out_i++) {
            std::size_t a_base = a->offset();
            std::size_t b_base = b->offset();

            // add batch offsets
            for (std::size_t dim = 0; dim < batch_ndim; dim++) {
                a_base += result_indices[dim] * this_batch_strides[dim];
                b_base += result_indices[dim] * other_batch_strides[dim];
            }

            // add row/col offsets
            if (this_dim >= 2) {
                a_base += result_indices[batch_ndim] * a_row_stride;
            }
            if (other_dim >= 2) {
                std::size_t col_index_pos = batch_ndim + (this_dim >= 2 ? 1 : 0);
                b_base += result_indices[col_index_pos] * b_col_stride;
            }

            float grad_val = out_grad->data().get()[out_i + out_grad->offset()];

            for (std::size_t k = 0; k < n; k++) {
                std::size_t a_idx = a_base + k * a_k_stride;
                std::size_t b_idx = b_base + k * b_k_stride;

                // map absolute data index to grad array index (grad array is contiguous, starting at offset 0)
                std::size_t a_grad_idx = a_idx - a->offset();
                std::size_t b_grad_idx = b_idx - b->offset();

                a_grad_data.get()[a_grad_idx] += grad_val * b->data().get()[b_idx];
                b_grad_data.get()[b_grad_idx] += grad_val * a->data().get()[a_idx];
            }

            // advance result indices
            if (!out_shape.empty()) {
                for (int dim = (int)(out_shape.size()) - 1; dim >= 0; dim--) {
                    result_indices[dim]++;
                    if (result_indices[dim] < out_shape[dim]) {
                        break;
                    }
                    result_indices[dim] = 0;
                }
            }
        }
    }
}
