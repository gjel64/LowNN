#include "core/grad_fn.hpp"

GradFn::GradFn(std::string name, std::vector<std::weak_ptr<Tensor>> inputs, std::weak_ptr<Tensor> output)
    : name(name), inputs(inputs), output(output)
{
}
