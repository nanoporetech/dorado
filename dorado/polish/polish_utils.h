#pragma once

#include <torch/torch.h>

#include <iosfwd>

namespace dorado::polisher {

void print_tensor_shape(std::ostream& os, const torch::Tensor& tensor);

}  // namespace dorado::polisher
