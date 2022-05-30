#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "MetalCRFModel.h"

#include "../utils/metal_utils.h"
#include "../utils/tensor_utils.h"

#include <math.h>
#include <toml.hpp>
#include <torch/torch.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

typedef uint16_t ftype;

static int get_gpu_core_count(MTL::Device *device) {
    std::string name = device->name()->utf8String();
    if (name == "Apple M1") {
        return 8;
    } else if (name == "Apple M1 Pro") {
        return 16;
    } else if (name == "Apple M1 Max") {
        return 32;
    } else if (name == "Apple M1 Ultra") {
        return 64;
    }
    return 8;
}

struct MetalLinearTanhImpl : Module {
    MetalLinearTanhImpl(int insize, int outsize) {
        auto weight = torch::empty({outsize, insize});
        auto bias = torch::empty({outsize});
        register_parameter("weight", weight, false);
        register_parameter("bias", bias, false);
    }
};

TORCH_MODULE(MetalLinearTanh);

struct MetalConv1dImpl : Module {
    MetalConv1dImpl(int layer,
                    int insize_,
                    int outsize_,
                    int k_,
                    int stride,
                    int chunk_size,
                    int batch_size,
                    MTL::Device *device)
            : insize(insize_), outsize(outsize_), k(k_) {
        assert(layer >= 1 && layer <= 3);

        if (layer == 1) {
            assert(outsize == 4);
            mat_weights = create_buffer(device, 49 * 8 * sizeof(ftype));
        } else if (layer == 2) {
            assert(outsize == 16);
            mat_weights = create_buffer(device, 29 * 16 * sizeof(ftype));
        } else {
            mat_weights = create_buffer(device, (k * insize + 1) * outsize * sizeof(ftype));
        }
        int32_t args_[] = {insize, k, outsize, stride, k / 2, chunk_size, batch_size};
        args = create_buffer(device, args_, 7);

        auto weight = torch::empty({outsize, insize, k});
        auto bias = torch::empty({outsize});

        register_parameter("weight", weight, false);
        register_parameter("bias", bias, false);

        conv_cps = make_cps(device, "conv" + std::to_string(layer) + "_simd");
        weights_cps = make_cps(device, "conv" + std::to_string(layer) + "_simd_reorder_weights");

        kernel_simd_groups = layer == 3 ? 4 : 16;
        kernel_thread_groups = get_gpu_core_count(device) * 4;
    }

    void run(MTL::CommandQueue *command_queue, MTL::Buffer *mat_in, MTL::Buffer *mat_out) {
        std::vector<MTL::Buffer *> buffers{args, mat_in, mat_weights, mat_out};
        launch_kernel(conv_cps, command_queue, buffers, kernel_thread_groups,
                      kernel_simd_groups * 32);
    }

    void run(MTL::CommandBuffer *command_buffer, MTL::Buffer *mat_in, MTL::Buffer *mat_out) {
        std::vector<MTL::Buffer *> buffers{args, mat_in, mat_weights, mat_out};
        launch_kernel_no_wait(conv_cps, command_buffer, buffers, kernel_thread_groups,
                              kernel_simd_groups * 32);
    }

    void load_weights(MTL::CommandQueue *command_queue) {
        auto params = named_parameters();
        auto t_w = *params.find("weight");
        auto t_b = *params.find("bias");
        t_w = t_w.permute({2, 1, 0}).contiguous();
        t_w = t_w.reshape({insize * k, outsize});
        t_w = torch::concat({t_w.flatten(0, -1), t_b}).contiguous();
        launch_kernel(weights_cps, command_queue, {args, mtl_for_tensor(t_w), mat_weights}, 1, 1);
    }

    MTL::Buffer *args, *mat_weights;
    MTL::ComputePipelineState *conv_cps, *weights_cps;
    int kernel_simd_groups, kernel_thread_groups;
    int insize, outsize, k;
};

TORCH_MODULE(MetalConv1d);

struct MetalLSTMImpl : Module {
    MetalLSTMImpl(int layer_size, bool reverse_, MTL::Device *device) : reverse(reverse_) {
        auto weight_ih = torch::empty({layer_size * 4, layer_size});
        auto weight_hh = torch::empty({layer_size * 4, layer_size});
        auto bias_ih = torch::empty({layer_size * 4});
        auto bias_hh = torch::empty({layer_size * 4});

        register_parameter("weight_ih", weight_ih, false);
        register_parameter("weight_hh", weight_hh, false);
        register_parameter("bias_ih", bias_ih, false);
        register_parameter("bias_hh", bias_hh, false);

        mat_weights = create_buffer(device,
                                    (size_t)(layer_size * 2 + 1) * layer_size * 4 * sizeof(ftype));
    }

    MTL::Buffer *mat_weights;
    bool reverse;
};

TORCH_MODULE(MetalLSTM);

struct MetalBlockImpl : Module {
    MetalBlockImpl(int chunk_size_,
                   int batch_size_,
                   int stride_,
                   int layer_size_,
                   int out_size_,
                   MTL::Device *device_)
            : device(device_),
              in_chunk_size(chunk_size_),
              stride(stride_),
              batch_size(batch_size_),
              layer_size(layer_size_),
              out_size(out_size_) {
        command_queue = device->newCommandQueue();

        constexpr int tile_size = 8;

        lstm_chunk_size = in_chunk_size / stride;

        // args for forward
        int32_t args_[] = {layer_size, 0, batch_size / tile_size, lstm_chunk_size, out_size};
        args[0] = create_buffer(device, args_, 5);
        // args for reverse
        args_[1] = 1;
        args[1] = create_buffer(device, args_, 5);
        // args for conversion to half
        args_[0] = in_chunk_size * batch_size;
        args[2] = create_buffer(device, args_, 1);

        switch (layer_size) {
        case 128:
            kernel_simd_groups = 16;
            break;
        case 192:
            kernel_simd_groups = 12;
            break;
        case 256:
            kernel_simd_groups = 32;
            break;
        case 384:
            kernel_simd_groups = 24;
            break;
        case 512:
            kernel_simd_groups = 32;
            break;
        default:
            kernel_simd_groups = 16;
        }
        kernel_thread_groups = get_gpu_core_count(device);

        fn[0] = "lstm_simd_" + std::to_string(layer_size) + "_fwd_" +
                std::to_string(kernel_simd_groups);
        fn[1] = "lstm_simd_" + std::to_string(layer_size) + "_rev_" +
                std::to_string(kernel_simd_groups);
        std::string linear_fn = "linear_tanh_simd_" + std::to_string(layer_size) + "_fwd_" +
                                std::to_string(kernel_simd_groups);
        lstm_cps[0] = make_cps(device, fn[0]);
        lstm_cps[1] = make_cps(device, fn[1]);
        linear_tanh_cps = make_cps(device, linear_fn);
        to_half_cps = make_cps(device, "float_to_half");
        reorder_input_cps = make_cps(device, "reorder_input");
        reorder_output_cps = make_cps(device, "reorder_output");
        reorder_weights_cps = make_cps(device, "reorder_weights");

        // TODO: we can reuse some of these matrices
        mat_transfer = create_buffer(
                device, (size_t)batch_size * lstm_chunk_size * out_size * sizeof(float));
        mat_transfer_ftype =
                (sizeof(ftype) == 4)
                        ? mat_transfer
                        : create_buffer(device, (size_t)batch_size * in_chunk_size * sizeof(ftype));
        mat_working_mem = create_buffer(
                device, size_t(lstm_chunk_size + 2) * batch_size * layer_size * sizeof(ftype));
        mat_state = create_buffer(device, batch_size * layer_size * sizeof(ftype));
        mat_temp_result = create_buffer(device, batch_size * layer_size * 4 * sizeof(ftype));

        conv1 = register_module("conv1",
                                MetalConv1d(1, 1, 4, 5, 1, in_chunk_size, batch_size, device));
        conv2 = register_module("conv2",
                                MetalConv1d(2, 4, 16, 5, 1, in_chunk_size, batch_size, device));
        conv3 = register_module("conv3", MetalConv1d(3, 16, layer_size, 19, stride, in_chunk_size,
                                                     batch_size, device));
        rnn1 = register_module("rnn_1", MetalLSTM(layer_size, true, device));
        rnn2 = register_module("rnn_2", MetalLSTM(layer_size, false, device));
        rnn3 = register_module("rnn_3", MetalLSTM(layer_size, true, device));
        rnn4 = register_module("rnn_4", MetalLSTM(layer_size, false, device));
        rnn5 = register_module("rnn_5", MetalLSTM(layer_size, true, device));
        linear = register_module("linear", MetalLinearTanh(layer_size, out_size));
    }

    void load_weights() {
        conv1->load_weights(command_queue);
        conv2->load_weights(command_queue);
        conv3->load_weights(command_queue);

        for (auto &rnn : {rnn1, rnn2, rnn3, rnn4, rnn5}) {
            auto params = rnn->named_parameters();

            auto t_w = *params.find("weight_ih");
            auto t_u = *params.find("weight_hh");
            auto t_b = *params.find("bias_ih");

            // reorder from IFGO to GIFO (2, 0, 1, 3)
            t_w = t_w.reshape({4, layer_size, layer_size}).transpose(1, 2);
            t_w = torch::concat({t_w[2], t_w[0], t_w[1], t_w[3]}, 1);

            t_u = t_u.reshape({4, layer_size, layer_size}).transpose(1, 2);
            t_u = torch::concat({t_u[2], t_u[0], t_u[1], t_u[3]}, 1);

            t_b = t_b.reshape({4, layer_size});
            t_b = torch::concat({t_b[2], t_b[0], t_b[1], t_b[3]});

            t_w = t_w.flatten(0, -1).contiguous();
            t_u = t_u.flatten(0, -1).contiguous();
            t_b = t_b.flatten(0, -1).contiguous();

            std::vector<MTL::Buffer *> buffers{args[rnn->reverse], mtl_for_tensor(t_w),
                                               mtl_for_tensor(t_u), mtl_for_tensor(t_b),
                                               rnn->mat_weights};
            launch_kernel(reorder_weights_cps, command_queue, buffers, kernel_thread_groups, 256);
        }

        auto params = linear->named_parameters();
        auto t_w = *params.find("weight");
        auto t_b = *params.find("bias");
        t_w = torch::concat({t_w.transpose(1, 0).contiguous().flatten(0, -1), t_b}).contiguous();

        if (sizeof(ftype) != sizeof(float)) {
            int numel = int(t_w.numel());
            MTL::Buffer *args = create_buffer(device, &numel, 1);
            mat_linear_weights = create_buffer(device, numel * sizeof(ftype));
            launch_kernel(to_half_cps, command_queue,
                          {args, mtl_for_tensor(t_w), mat_linear_weights}, kernel_thread_groups,
                          256);
        } else {
            mat_linear_weights = extract_mtl_from_tensor(t_w);
        }
    }

    std::pair<MTL::CommandBuffer *, torch::Tensor> forward_async(torch::Tensor x, bool lock_gpu) {
        auto command_buffer = command_queue->commandBuffer();

        if (sizeof(ftype) == 2) {
            launch_kernel_no_wait(to_half_cps, command_buffer,
                                  {args[2], mtl_for_tensor(x), mat_transfer_ftype},
                                  kernel_thread_groups, 256);
            conv1->run(command_buffer, mat_transfer_ftype, mat_working_mem);
        } else {
            conv1->run(command_buffer, mtl_for_tensor(x), mat_working_mem);
        }
        conv2->run(command_buffer, mat_working_mem, mat_transfer);
        conv3->run(command_buffer, mat_transfer, mat_working_mem);

        for (auto &rnn : {rnn1, rnn2, rnn3, rnn4, rnn5}) {
            std::vector<MTL::Buffer *> buffers{args[rnn->reverse], mat_working_mem,
                                               rnn->mat_weights, mat_state, mat_temp_result};
            launch_kernel_no_wait(lstm_cps[rnn->reverse], command_buffer, buffers,
                                  kernel_thread_groups, kernel_simd_groups * 32);
        }

        x = torch::empty({lstm_chunk_size, batch_size, out_size});
        launch_kernel_no_wait(linear_tanh_cps, command_buffer,
                              {args[0], mat_working_mem, mat_linear_weights, mtl_for_tensor(x)},
                              kernel_thread_groups, kernel_simd_groups * 32);

        // TODO: Find a better way of dealing with long-running kernels.
        // This is a hacky way of avoiding Metal kernel launch timeouts. We only let one runner thread
        // access the GPU. Unlock happens in MTLDecoder::beam_search. We want one thread to finish work on the GPU
        // and start CPU decoding before another thread starts GPU work.
        if (lock_gpu) {
            lock_mtl_device();
        }
        command_buffer->commit();
        return std::make_pair(command_buffer, x);
    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor out;
        // TODO: find a more robust way of dealing with Metal kernel launch issues
        for (int retry = 0; retry < 5; ++retry) {
            MTL::CommandBuffer *command_buffer;
            std::tie(command_buffer, out) = forward_async(x, retry == 0);
            command_buffer->waitUntilCompleted();
            if (command_buffer->status() == MTL::CommandBufferStatusCompleted) {
                break;
            }
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(20ms);
        }
        return out;
    }

    MTL::Device *device;
    MTL::CommandQueue *command_queue;
    std::string fn[2];
    MTL::ComputePipelineState *reorder_weights_cps, *reorder_input_cps, *reorder_output_cps,
            *lstm_cps[2], *to_half_cps, *linear_tanh_cps;
    MTL::Buffer *mat_transfer, *mat_transfer_ftype, *args[3], *mat_working_mem, *mat_state,
            *mat_temp_result, *mat_linear_weights;
    int in_chunk_size, lstm_chunk_size, stride, batch_size, layer_size, out_size,
            kernel_thread_groups, kernel_simd_groups;
    MetalLSTM rnn1{nullptr}, rnn2{nullptr}, rnn3{nullptr}, rnn4{nullptr}, rnn5{nullptr};
    MetalConv1d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    MetalLinearTanh linear{nullptr};
};

TORCH_MODULE(MetalBlock);

struct MetalModelImpl : Module {
    MetalModelImpl(int size,
                   int outsize,
                   int stride,
                   int chunk_size,
                   int batch_size,
                   MTL::Device *device) {
        mtl_block = register_module(
                "mtl_block", MetalBlock(chunk_size, batch_size, stride, size, outsize, device));
    }

    void load_state_dict(std::vector<torch::Tensor> weights) {
        assert(weights.size() == parameters().size());
        for (size_t idx = 0; idx < weights.size(); idx++) {
            parameters()[idx].data() = weights[idx].data();
        }

        mtl_block->load_weights();
    }

    torch::Tensor forward(torch::Tensor x) { return mtl_block->forward(x); }

    MetalBlock mtl_block{nullptr};
};

TORCH_MODULE(MetalModel);

ModuleHolder<AnyModule> load_crf_mtl_model(const std::string &path,
                                           int batch_size,
                                           int chunk_size,
                                           torch::TensorOptions options) {
    auto device = get_mtl_device();

    auto config = toml::parse(path + "/config.toml");

    const auto &encoder = toml::find(config, "encoder");
    const auto scale = toml::find<float>(encoder, "scale");
    const auto stride = toml::find<int>(encoder, "stride");
    const auto insize = toml::find<int>(encoder, "features");
    const auto blank_score = toml::find<float>(encoder, "blank_score");

    const auto &global_norm = toml::find(config, "global_norm");
    const auto state_len = toml::find<int>(global_norm, "state_len");

    int states = pow(4, state_len);
    int outsize = states * 5;

    auto state_dict = load_weights(path);

    auto lw = state_dict[state_dict.size() - 2];
    auto lb = state_dict[state_dict.size() - 1];

    state_dict[state_dict.size() - 2] =
            F::pad(lw.view({states, 4, insize}), F::PadFuncOptions({0, 0, 1, 0}).value(0.0))
                    .view({outsize, insize});

    state_dict[state_dict.size() - 1] =
            F::pad(lb.view({states, 4}),
                   F::PadFuncOptions({1, 0}).value(atanh(blank_score / scale)))
                    .view({outsize});

    auto model = MetalModel(insize, outsize, stride, chunk_size, batch_size, device);
    model->load_state_dict(state_dict);
    model->eval();

    auto module = AnyModule(model);
    auto holder = ModuleHolder<AnyModule>(module);

    return holder;
}
