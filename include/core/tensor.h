#include <vector>
#include <memory>
/*
TODO:
    - -1 indexing
*/

class Tensor {
private:
    std::vector<float> _data;
    std::vector<std::size_t> _shape;
    std::vector<std::size_t> _strides;
    std::size_t _size;

public:
    // -------- Init --------
    Tensor(float data);
    Tensor(std::vector<float> data);
    Tensor(std::vector<std::vector<float>> data);
    Tensor(std::vector<std::size_t> shape);

    // -------- Getters/Setters --------
    float item() const;
    std::shared_ptr<Tensor> operator[](int index);
    std::shared_ptr<Tensor> operator[](std::vector<int> indices);
    std::vector<std::size_t> shape() const;
    std::vector<std::size_t> strides() const;
    std::size_t size() const;


};