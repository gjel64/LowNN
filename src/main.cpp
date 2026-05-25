#include <iostream>
#include <vector>
#include "core/tensor.hpp"

int main() {
    std::cout << "Hello, LowNN!" << std::endl;

    std::vector<float> test = {1.0f};
    std::vector<float> tr_test = {test.begin(), test.end() - 2};
    
    std::cout << tr_test.size() << std::endl;
    return 0;
}