#include <iostream>
#include <vector>
#include "core/tensor.hpp"

int main() {
    std::cout << "Hello, LowNN!" << std::endl;

    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::cout << (*data.begin()) << " " << (*(data.end() - 1)) << "\n";


    /*
    std::vector<std::vector<float>> data(5, std::vector<float>(3, 1.0f));
    std::cout << "Data: \n[ \n";
    std::string tab = "";
    for (auto& row : data) 
    {
        tab = "   ";
        std::cout << tab.c_str() << "[ ";
        for (auto& val : row) 
        {
            std::cout << val << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << "]\n";
    */
    return 0;
}