#pragma once

#include <torch/nn.h>

namespace dorado::basecall::nn {
std::string shape(const at::Tensor &t, const std::string &name);
void dump_tensor(const at::Tensor &t, const std::string &name);
}  // namespace dorado::basecall::nn