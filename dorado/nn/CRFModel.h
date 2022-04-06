#ifndef DORADO_CRFMODEL_H
#define DORADO_CRFMODEL_H

#include <torch/torch.h>

torch::nn::ModuleHolder<torch::nn::AnyModule> load_crf_model(std::string path, int batch_size, int chunk_size, torch::TensorOptions options);

#endif  // DORADO_CRFMODEL_H
