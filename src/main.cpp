#include <iostream>
#include <vector>
#include "core/tensor.hpp"

int main() {
    std::cout << "Hello, LowNN!" << std::endl;

    std::vector<std::size_t> a = {1, 2, 3};
    auto b = a;
    b[0] = 2;
    std::cout << "a : [" << a[0] << a[1] << a[2] << "]";

    return 0;
}