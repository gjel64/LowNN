#include <iostream>
#include <vector>
#include "core/tensor.hpp"

int main() {
    std::cout << "Hello, LowNN!" << std::endl;

    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(std::vector<float>{{1.0f, 2.0f}}, true);
    std::shared_ptr<Tensor> t2 = (*t1)[0];
    t2->backward();

    return 0;
}