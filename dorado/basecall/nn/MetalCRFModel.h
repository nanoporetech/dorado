#pragma once

#include "config/BasecallModelConfig.h"
#include "torch_utils/metal_utils.h"

#include <torch/nn.h>

#include <vector>

namespace dorado::basecall::nn {

struct MetalLinearImpl : torch::nn::Module {
    MetalLinearImpl(int insize, int outsize, bool has_bias);
};

TORCH_MODULE(MetalLinear);

struct MetalConv1dImpl : torch::nn::Module {
    MetalConv1dImpl(int layer,
                    int in_size_,
                    int out_size_,
                    int win_size_,
                    int stride_,
                    config::Activation activation,
                    int chunk_size_,
                    int batch_size_,
                    MTL::Device *const device);

    void run(MTL::CommandQueue *command_queue, MTL::Buffer *mat_in, MTL::Buffer *mat_out);
    void run(MTL::CommandBuffer *command_buffer, MTL::Buffer *mat_in, MTL::Buffer *mat_out);

    void load_weights();

    at::Tensor t_weights_bias;
    std::vector<NS::SharedPtr<MTL::Buffer>> m_args;
    NS::SharedPtr<MTL::ComputePipelineState> conv_cps, weights_cps;
    int kernel_simd_groups, kernel_thread_groups;
    int in_size, out_size, win_size, stride, chunk_size, batch_size, w_pad_rows, repeats;
};

TORCH_MODULE(MetalConv1d);

struct MetalLSTMImpl : torch::nn::Module {
    MetalLSTMImpl(int layer_size, bool reverse_);
    at::Tensor t_weights_bias;
    bool reverse;
};

TORCH_MODULE(MetalLSTM);

struct MetalBlockImpl : torch::nn::Module {
    MetalBlockImpl(int chunk_size_,
                   int batch_size_,
                   const config::BasecallModelConfig &config_,
                   int out_split_,
                   MTL::Device *const device);

    void load_weights();

    // Executes the model, with the linear layer held off by linear_hold_off, if non-NULL.
    // If CB submissions are successful, it returns the command buffer used for the linear layer
    // and scan kernels.  If either CB is unsuccessful, it returns nullptr.
    MTL::CommandBuffer *forward_async(at::Tensor &in,
                                      MTL::SharedEvent *const linear_hold_off_event,
                                      uint64_t linear_hold_off_id,
                                      int try_count,
                                      std::vector<at::Tensor> &out);

    MTL::Device *m_device;
    NS::SharedPtr<MTL::CommandQueue> m_command_queue;
    NS::SharedPtr<MTL::ComputePipelineState> lstm_cps[2], linear_cps[2];
    NS::SharedPtr<MTL::Buffer> mat_working_mem, mat_state, mat_temp, linear_weights[2],
            args_linear2;
    // Each args buffer corresponds to a different time span of the LSTM layer.
    std::vector<NS::SharedPtr<MTL::Buffer>> m_args_lstm;
    std::vector<NS::SharedPtr<MTL::Buffer>> args_linear;
    int in_chunk_size, lstm_chunk_size, batch_size, kernel_thread_groups, kernel_simd_groups;
    config::BasecallModelConfig config;
    MetalLSTM rnn1{nullptr}, rnn2{nullptr}, rnn3{nullptr}, rnn4{nullptr}, rnn5{nullptr};
    MetalConv1d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    MetalLinear linear1{nullptr}, linear2{nullptr};
};

TORCH_MODULE(MetalBlock);

struct MetalCRFModelImpl : torch::nn::Module {
    MetalCRFModelImpl(const config::BasecallModelConfig &config,
                      int chunk_size,
                      int batch_size,
                      int out_split,
                      MTL::Device *const device);

    void load_state_dict(const std::vector<at::Tensor> &weights);

    MTL::CommandBuffer *forward_async(at::Tensor &in,
                                      MTL::SharedEvent *const linear_hold_off_event,
                                      uint64_t linear_hold_off_id,
                                      int try_count,
                                      std::vector<at::Tensor> &out);
    MetalBlock mtl_block{nullptr};
};

TORCH_MODULE(MetalCRFModel);

}  // namespace dorado::basecall::nn
