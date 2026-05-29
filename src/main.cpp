#include <iostream>
#include <vector>
#include "core/tensor.hpp"

int main() {
    std::cout << "Hello, LowNN!" << std::endl;

    std::shared_ptr<Tensor> t = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    std::cout << "Tensor t1:" << std::endl;
    t->print();
    std::cout << "\n";
    (*t)[1]->print();

    return 0;
}