#include "core/engine.hpp"
#include "core/tensor.hpp"


void Engine::backward(std::shared_ptr<Tensor> root, std::shared_ptr<Tensor> grad_override) {
    if (!root) return;

    std::shared_ptr<float[]> root_grad_data = std::make_shared<float[]>(root->size());

    // initialize root grad
    if (grad_override) {
        if (grad_override->size() != root->size()) {
            throw std::runtime_error("Engine::backward: size mismatch between grad_override and root");
        }
        // overwrite the root's grad (accumulate at the end) (because this is the loss that backprop on the graph and not the roots tensors grad)
        for (std::size_t i = 0; i < root->size(); ++i) {
            root_grad_data.get()[i] = root->gradp().get()[i];
            root->gradp().get()[i] = grad_override->data().get()[i + grad_override->offset()];
        }
    } 
    else {
        if (root->size() == 1) {
            root->ensure_grad_allocated();
            root->gradp().get()[0] += 1.0f;
        } 
        else {
            throw std::invalid_argument("Engine::backward: grad_override parameter must be provided for non-scalar root");
        }
    }

    // collect grad fns reachable (postorder)
    std::vector<std::shared_ptr<GradFn>> topo;
    std::unordered_set<GradFn*> visited; // pointer identity is fine for uniqueness

    std::function<void(std::shared_ptr<GradFn>)> dfs = [&](std::shared_ptr<GradFn> fn){
        if (!fn) return;
        if (visited.count(fn.get())) return;
        visited.insert(fn.get());
        for (auto &w_in : fn->inputs) {
            if (auto in = w_in.lock()) {
                    if (in->grad_fn()) dfs(in->grad_fn());
            }
        }
        topo.push_back(fn);
    };

    if (root->grad_fn()) dfs(root->grad_fn()); // builds topo

    // execute in reverse topo: parents before children
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        auto fn = *it;
        auto out = fn->output.lock();
        if (!out) continue;

        std::shared_ptr<Tensor> out_grad = nullptr;
        if (out->gradp()) {
            out_grad = std::make_shared<Tensor>(out->gradp(), out->shape(), 0, out->size(), false);
        } else {
            throw std::runtime_error("Engine::backward: missing out's grad");
        }

        fn->backward(out, out_grad);
    }

    // the accumulation as i told you before : (ugly for now)
    if (grad_override) {
        for (std::size_t i = 0; i < root->size(); ++i) {
            root->gradp().get()[i] += root_grad_data.get()[i];
        }
    }
}
