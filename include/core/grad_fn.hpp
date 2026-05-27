#pragma once
#include <string>
#include <functional>

class Tensor;

class GradFn {
private:
    std::function<void(Tensor&)> _fn;
    std::string _name;
public:
    GradFn(std::string name);
    void operator()(Tensor& t) const { _fn(t); }
    std::string name() const { return _name; }
};