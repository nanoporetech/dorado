#include <torch/torch.h>

void serialise_tensor(torch::Tensor t, std::string path);
std::vector<torch::Tensor> load_weights(std::string dir);