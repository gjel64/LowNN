#include <iostream>
#include <vector>
#include "core/tensor.hpp"

int main() {
    std::cout << "Hello, LowNN!" << std::endl;

    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    Tensor(data).print();
    

    return 0;
}