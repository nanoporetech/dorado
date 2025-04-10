#pragma once

#include "WorkingMemory.h"
#include "config/common.h"

#include <torch/nn.h>

namespace dorado::nn {

struct ConvStackImpl : torch::nn::Module {
    explicit ConvStackImpl(const std::vector<config::ConvParams> &layer_params);
#if DORADO_CUDA_BUILD
    void reserve_working_memory(WorkingMemory &wm, std::optional<TensorLayout> output_layout);
    void run_koi(WorkingMemory &wm);
#endif  // if DORADO_CUDA_BUILD

    at::Tensor forward(at::Tensor x);

    struct ConvLayer {
        explicit ConvLayer(const config::ConvParams &params);
        const config::ConvParams params;
        torch::nn::Conv1d conv{nullptr};
#if DORADO_CUDA_BUILD
        TensorLayout output_layout{TensorLayout::NTC};
        bool cutlass_conv{false};
        int output_T_padding{0};
        at::Tensor w_device;
        at::Tensor b_device;

        void reserve_working_memory(WorkingMemory &wm);
        void run_koi(WorkingMemory &wm);
#endif  // if DORADO_CUDA_BUILD
    };

    std::vector<ConvLayer> layers;
};

TORCH_MODULE(ConvStack);

}  // namespace dorado::nn