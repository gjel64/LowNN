#pragma once
#include <memory>
#include <vector>
#include <unordered_set>
#include <functional>

class Tensor;

class Engine {
public:
    static void backward(std::shared_ptr<Tensor> root, std::shared_ptr<Tensor> grad_override = nullptr);
};
