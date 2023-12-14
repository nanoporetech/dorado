#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "MetalCRFModel.h"

#include "CRFModelConfig.h"
#include "crf_utils.h"
#include "decode/beam_search.h"
#include "utils/math_utils.h"
#include "utils/metal_utils.h"
#include "utils/module_utils.h"
#include "utils/tensor_utils.h"

#include <math.h>
#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <chrono>
#include <set>
#include <vector>

using namespace dorado::utils;
using namespace std::chrono_literals;
using namespace torch::nn;
namespace F = torch::nn::functional;
using torch::indexing::Ellipsis;
using torch::indexing::Slice;

static constexpr auto torch_dtype = torch::kF16;
static const size_t dtype_bytes = torch::elementSize(torch_dtype);

namespace MTL {
auto format_as(CommandBufferStatus status) { return fmt::underlying(status); }
}  // namespace MTL

namespace {
// SIMD tile size dictated by the Metal spec.
constexpr int kTileSize = 8;
// We assume non-AMD GPUs, in which case this is 32.
constexpr int kSIMDGroupWidth = 32;

// Returns true on success.
bool finishCommandBuffer(std::string_view label, MTL::CommandBuffer *cb, int try_count) {
    cb->commit();
    cb->waitUntilCompleted();

    auto status = cb->status();
    bool success = (status == MTL::CommandBufferStatusCompleted);
    if (success) {
        spdlog::debug("Metal command buffer {}: {} GPU ms {} CPU ms succeeded (try {})", label,
                      1000.f * float(cb->GPUEndTime() - cb->GPUStartTime()),
                      1000.f * float(cb->kernelEndTime() - cb->kernelStartTime()), try_count);
    } else {
        spdlog::warn("Metal command buffer {} failed: status {} (try {})", label, status,
                     try_count);
        if (status == MTL::CommandBufferStatusError) {
            const auto *const error_ptr = cb->error();
            if (error_ptr)
                spdlog::warn("Command buffer error code: {} ({})", error_ptr->code(),
                             error_ptr->localizedDescription()->utf8String());
        }
    }
    return success;
}

}  // namespace

namespace dorado::basecall {

namespace nn {

struct MetalLinearImpl : Module {
    MetalLinearImpl(int insize, int outsize, bool has_bias) {
        auto weight = torch::empty({outsize, insize});
        auto bias = torch::empty({outsize});
        register_parameter("weight", weight, false);
        if (has_bias)
            register_parameter("bias", bias, false);
    }
};

TORCH_MODULE(MetalLinear);

struct MetalConv1dImpl : Module {
    MetalConv1dImpl(int layer,
                    int in_size_,
                    int out_size_,
                    int win_size_,
                    int stride_,
                    Activation activation,
                    int chunk_size_,
                    int batch_size_,
                    MTL::Device *const device)
            : in_size(in_size_),
              out_size(out_size_),
              win_size(win_size_),
              stride(stride_),
              chunk_size(chunk_size_),
              batch_size(batch_size_) {
        assert(layer >= 1 && layer <= 3);

        // For layers 1 and 2 we only have kernels for particular in/out feature sizes.
        if (layer == 1) {
            // Simplex in_size == 1.  Duplex in_size == 13.
            assert(in_size == 1 || in_size == 13);
            assert(out_size == 4 || out_size == 16);
            assert(win_size = 5);
        } else if (layer == 2) {
            assert(in_size == 4 || in_size == 16);
            assert(out_size == 16);
            assert(win_size = 5);
        }

        if (layer != 3) {
            // Layers 1 and 2 are executed with a single kernel launch.
            // The last 2 arguments are unused.
            const std::vector<int32_t> args{in_size,    win_size,   out_size, stride, win_size / 2,
                                            chunk_size, batch_size, 0,        0};
            m_args.push_back(create_vec_buffer(device, args));
        } else {
            // We cut up the time span for individual kernel launches for conv3 since it is by far
            // the most time consuming, and sup times can be of the order of seconds, which
            // is known to provoke command buffer errors.
            // The last 2 arguments specify the output time step range, i.e. time step range after
            // dividing by stride.
            const int output_time_step_count = chunk_size / stride;
            constexpr int kMaxTimeSteps = 20;
            const int num_pieces = (output_time_step_count + kMaxTimeSteps - 1) / kMaxTimeSteps;
            for (int i = 0; i < num_pieces; ++i) {
                const int time_step_begin = i * kMaxTimeSteps;
                const int time_step_end = std::min((i + 1) * kMaxTimeSteps, output_time_step_count);
                const std::vector<int32_t> args{in_size,    win_size,        out_size,
                                                stride,     win_size / 2,    chunk_size,
                                                batch_size, time_step_begin, time_step_end};
                m_args.push_back(create_vec_buffer(device, args));
            }
            spdlog::debug("conv3 output_time_step_count {} => {} kernel launches",
                          output_time_step_count, num_pieces);
        }

        auto weight = torch::empty({out_size, in_size, win_size}, torch::kF32);
        auto bias = torch::empty({out_size}, torch::kF32);
        register_parameter("weight", weight, false);
        register_parameter("bias", bias, false);

        // As we use simdgroup matrix math with a tile size of 8, we need to
        // - Expand convs with out_size == 4 so that out_size is effectively 8 by repeating the weights
        //   ({in=1, win=5, out=4, stride=1} becomes effectively {in=1, win=6 out=8, stride=2}).
        // - Add zero-pad rows around the weight matrix in case the number of rows taking part in
        //   matrix multiplication is not a multiple of 8. To deal with the chunk edges, we need
        //   the non-zero rows either at the top or bottom of the tile, so we add padding on both sides.
        repeats = 1;
        w_pad_rows = 0;
        if (in_size == 1 && out_size == 4) {
            repeats = 2;     // expand to out_size 8
            w_pad_rows = 4;  // At the edges we only use 4 weight rows, need to pad to 8
        } else if (in_size == 1 && out_size == 16) {
            w_pad_rows = 5;  // At the edges we only use 3 weight rows, need to pad to 8
        } else if (in_size == 4 && out_size == 16) {
            w_pad_rows = 4;  // Matrix has 20 rows, need to pad to 24
        }
        int new_out_size = repeats * out_size;
        int rows = 2 * w_pad_rows + (win_size + (repeats - 1) * stride) * in_size + 1;
        t_weights_bias = torch::zeros({rows, new_out_size}, torch_dtype);

        kernel_simd_groups = (layer == 3 || (layer == 2 && in_size == 16)) ? 4 : 16;
        kernel_thread_groups = get_mtl_device_core_count() * 4;

        std::vector<std::tuple<std::string, MetalConstant>> metal_constants = {
                {"kConvOutputClamp", activation == Activation::SWISH_CLAMP},
                {"kConvTanhActivation", activation == Activation::TANH}};
        const int kernel_threads = kSIMDGroupWidth * kernel_simd_groups;
        std::string kernel_name = "conv" + std::to_string(layer);
        // Layer 1 and 2 conv kernels are tailored to specific feature sizes.
        if (layer == 1 || layer == 2) {
            kernel_name += "_in" + std::to_string(in_size) + "_out" + std::to_string(out_size);
        }

        conv_cps = make_cps(device, kernel_name + "_simd", metal_constants, kernel_threads);
    }

    void run(MTL::CommandQueue *command_queue, MTL::Buffer *mat_in, MTL::Buffer *mat_out) {
        for (const auto &args : m_args) {
            std::vector<MTL::Buffer *> buffers{args.get(), mat_in, mtl_for_tensor(t_weights_bias),
                                               mat_out};
            launch_kernel(conv_cps.get(), command_queue, buffers, {}, kernel_thread_groups,
                          kernel_simd_groups * kSIMDGroupWidth);
        }
    }

    void run(MTL::CommandBuffer *command_buffer, MTL::Buffer *mat_in, MTL::Buffer *mat_out) {
        for (const auto &args : m_args) {
            std::vector<MTL::Buffer *> buffers{args.get(), mat_in, mtl_for_tensor(t_weights_bias),
                                               mat_out};
            launch_kernel_no_wait(conv_cps.get(), command_buffer, buffers, {}, kernel_thread_groups,
                                  kernel_simd_groups * kSIMDGroupWidth);
        }
    }

    void load_weights() {
        auto params = named_parameters();
        auto t_w = *params.find("weight");
        auto t_b = *params.find("bias");
        t_w = t_w.permute({2, 1, 0});

        for (int i = 0; i < repeats; ++i) {
            int out_row = w_pad_rows + i * stride * in_size;
            t_weights_bias
                    .index({Slice(out_row, out_row + win_size * in_size),
                            Slice(i * out_size, (i + 1) * out_size)})
                    .view({win_size, in_size, out_size}) = t_w;
            t_weights_bias.index({-1, Slice(i * out_size, (i + 1) * out_size)}) = t_b;
        }
    }

    at::Tensor t_weights_bias;
    std::vector<NS::SharedPtr<MTL::Buffer>> m_args;
    NS::SharedPtr<MTL::ComputePipelineState> conv_cps, weights_cps;
    int kernel_simd_groups, kernel_thread_groups;
    int in_size, out_size, win_size, stride, chunk_size, batch_size, w_pad_rows, repeats;
};

TORCH_MODULE(MetalConv1d);

static constexpr int kLstmGates = 4;
struct MetalLSTMImpl : Module {
    MetalLSTMImpl(int layer_size, bool reverse_) : reverse(reverse_) {
        auto weight_ih = torch::empty({layer_size * kLstmGates, layer_size});
        auto weight_hh = torch::empty({layer_size * kLstmGates, layer_size});
        auto bias_ih = torch::empty({layer_size * kLstmGates});
        auto bias_hh = torch::empty({layer_size * kLstmGates});

        register_parameter("weight_ih", weight_ih, false);
        register_parameter("weight_hh", weight_hh, false);
        register_parameter("bias_ih", bias_ih, false);
        register_parameter("bias_hh", bias_hh, false);

        // For non-obvious reasons the LSTM kernel runs faster if the U (or _hh) and W (or _ih) matrices are
        // spaced such that there is room for one more matrix between them. Thus a factor of 3 instead of 2.
        t_weights_bias = torch::empty({layer_size * 3 + 1, layer_size, kLstmGates}, torch_dtype);
    }

    at::Tensor t_weights_bias;
    bool reverse;
};

TORCH_MODULE(MetalLSTM);

struct MetalBlockImpl : Module {
    MetalBlockImpl(int chunk_size_,
                   int batch_size_,
                   const CRFModelConfig &config_,
                   int out_split_,
                   MTL::Device *const device)
            : m_device(device),
              in_chunk_size(chunk_size_),
              batch_size(batch_size_),
              config(config_) {
        m_command_queue = NS::TransferPtr(m_device->newCommandQueue());

        lstm_chunk_size = in_chunk_size / config.stride;

        // args for LSTM kernel
        {
            // We enforce a maximum time step count for each LSTM kernel because long running
            // kernels increase the likelihood of command buffer submission errors, of various
            // types.  Each args buffer here is for a different time step range.
            constexpr int kMaxTimeSteps = 20;
            const int num_pieces = (lstm_chunk_size + kMaxTimeSteps - 1) / kMaxTimeSteps;
            for (int i = 0; i < num_pieces; ++i) {
                const int time_step_begin = i * kMaxTimeSteps;
                const int time_step_end = std::min((i + 1) * kMaxTimeSteps, lstm_chunk_size);
                std::vector<int32_t> args{batch_size / kTileSize, lstm_chunk_size, time_step_begin,
                                          time_step_end};
                auto args_buffer = create_vec_buffer(m_device, args);
                m_args_lstm.push_back(args_buffer);
            }
            spdlog::debug("lstm_chunk_size {} => {} LSTM kernel launches", lstm_chunk_size,
                          num_pieces);
        }

        // args for final (possibly only) linear layer kernel.
        // Each output buffer requires a distinct input offset, so we must have a separate args buffer.
        args_linear.resize(out_split_);
        const int32_t in_batch_tiles = batch_size / kTileSize;
        const int32_t out_batch_tiles = (batch_size / out_split_) / kTileSize;
        for (int i = 0; i < out_split_; ++i) {
            const int32_t in_batch_tile_offset = out_batch_tiles * i;
            std::vector<int32_t> args_linear_ = {in_batch_tiles, in_batch_tile_offset,
                                                 out_batch_tiles, lstm_chunk_size};
            args_linear.at(i) = create_vec_buffer(m_device, args_linear_);
        }
        args_linear2 = create_vec_buffer<int32_t>(
                device, {out_batch_tiles, 0, out_batch_tiles, lstm_chunk_size});

        switch (config.lstm_size) {
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
        case 768:
            kernel_simd_groups = 32;
            break;
        case 1024:
            kernel_simd_groups = 32;
            break;
        default:
            kernel_simd_groups = 16;
        }
        kernel_thread_groups = get_mtl_device_core_count();
        const int lstm_threads = kernel_simd_groups * kSIMDGroupWidth;
        lstm_cps[0] =
                make_cps(m_device, "lstm",
                         {{"kLstmLayerSize", config.lstm_size}, {"kLstmReversedInTime", false}},
                         lstm_threads);
        lstm_cps[1] =
                make_cps(m_device, "lstm",
                         {{"kLstmLayerSize", config.lstm_size}, {"kLstmReversedInTime", true}},
                         lstm_threads);

        // The temp buffer used for these purposes (number of elements of `torch_dtype` in []):
        // - Store output of second conv layer [in_chunk_size * batch_size * kMaxConv2OutChannels]
        // - Store output of first linear layer if there are two
        //   [lstm_chunk_size * batch_size * decomposition / out_split_]
        // We size mat_temp here for conv2 output, potentially increasing it below in the case of a
        // linear decomposition model.
        constexpr int kMaxConv2OutChannels = 16;
        int mat_temp_elems = batch_size * kMaxConv2OutChannels * in_chunk_size;

        conv1 = register_module("conv1", MetalConv1d(1, config.num_features, config.convs[0].size,
                                                     5, 1, config.convs[0].activation,
                                                     in_chunk_size, batch_size, device));
        conv2 = register_module(
                "conv2", MetalConv1d(2, config.convs[0].size, 16, 5, 1, config.convs[1].activation,
                                     in_chunk_size, batch_size, device));
        conv3 = register_module("conv3", MetalConv1d(3, 16, config.lstm_size, 19, config.stride,
                                                     config.convs[2].activation, in_chunk_size,
                                                     batch_size, device));
        rnn1 = register_module("rnn_1", MetalLSTM(config.lstm_size, true));
        rnn2 = register_module("rnn_2", MetalLSTM(config.lstm_size, false));
        rnn3 = register_module("rnn_3", MetalLSTM(config.lstm_size, true));
        rnn4 = register_module("rnn_4", MetalLSTM(config.lstm_size, false));
        rnn5 = register_module("rnn_5", MetalLSTM(config.lstm_size, true));

        const int linear_threads = kernel_simd_groups * kSIMDGroupWidth;
        // If the intermediate feature size between conv1 and conv2 is 16, then this is a v4
        // type model, where the linear layer output is clamped rather than run through tanh.
        // Otherwise the intermediate feature size is 4.

        if (config.out_features.has_value()) {
            // The linear layer is decomposed into 2 matmuls.
            const int decomposition = config.out_features.value();
            linear1 = register_module("linear1",
                                      MetalLinear(config.lstm_size, decomposition, config.bias));
            const bool kSecondLayerBias = false;
            linear2 = register_module("linear2",
                                      MetalLinear(decomposition, config.outsize, kSecondLayerBias));
            const auto linear_constants1 = std::vector<std::tuple<std::string, MetalConstant>>(
                    {{"kLinearInSize", config.lstm_size},
                     {"kLinearOutSize", decomposition},
                     {"kLinearOutputScale", 1.0f},
                     {"kLinearOutputClamp", false},
                     {"kLinearOutputTanh", false},
                     {"kLinearOutputAsByte", false}});
            linear_cps[0] =
                    make_cps(m_device, "linear_from_rev_lstm", linear_constants1, linear_threads);
            const auto linear_constants2 = std::vector<std::tuple<std::string, MetalConstant>>(
                    {{"kLinearInSize", decomposition},
                     {"kLinearOutSize", config.outsize},
                     // Rescale from clamped [-5.0, 5.0] range to byte range.
                     {"kLinearOutputScale", 127.0f / 5.0f},
                     {"kLinearOutputClamp", true},
                     {"kLinearOutputTanh", false},
                     {"kLinearOutputAsByte", true}});
            linear_cps[1] = make_cps(m_device, "linear", linear_constants2, linear_threads);
            // We use mat_temp for the output of the first linear layer, so ensure it is large
            // enough for that purpose.
            mat_temp_elems = std::max(mat_temp_elems,
                                      decomposition * (batch_size / out_split_) * lstm_chunk_size);
        } else {
            const bool is_v3_model = (config.num_features == 1 && config.convs[0].size == 4) ||
                                     (config.num_features == 13 && config.convs[0].size == 16);
            const auto linear_constants = std::vector<std::tuple<std::string, MetalConstant>>(
                    {{"kLinearInSize", config.lstm_size},
                     {"kLinearOutSize", config.outsize},
                     // If v4, rescale from clamped [-5.0, 5.0] range to byte range.
                     // If v3, rescale from tanh [-1.0, 1,0] range to byte range.
                     {"kLinearOutputScale", is_v3_model ? 127.0f : (127.0f / 5.0f)},
                     {"kLinearOutputClamp", !is_v3_model},
                     {"kLinearOutputTanh", is_v3_model},
                     {"kLinearOutputAsByte", true}});
            linear_cps[0] =
                    make_cps(m_device, "linear_from_rev_lstm", linear_constants, linear_threads);
            // Single matmul that may or may not have a bias.
            if (!config.out_features.has_value()) {
                linear1 = register_module(
                        "linear1", MetalLinear(config.lstm_size, config.outsize, config.bias));
            }
        }

        // This buffer is used for several layers of the model.
        mat_working_mem = create_buffer(m_device, size_t(lstm_chunk_size + 3) * batch_size *
                                                          config.lstm_size * dtype_bytes);
        mat_state = create_buffer(m_device, batch_size * config.lstm_size * dtype_bytes);
        mat_temp = create_buffer(m_device, mat_temp_elems * dtype_bytes);
    }

    void load_weights() {
        conv1->load_weights();
        conv2->load_weights();
        conv3->load_weights();

        for (auto &&rnn : {rnn1, rnn2, rnn3, rnn4, rnn5}) {
            auto params = rnn->named_parameters();

            auto t_w = *params.find("weight_ih");
            auto t_u = *params.find("weight_hh");
            auto t_b = *params.find("bias_ih");

            // The effect of time reversal is accommodated by swapping buffer order.
            if (rnn->reverse) {
                std::swap(t_w, t_u);
            }

            // Reshape and combine matrices into one of size {kLstmGates, 3 * layer_size + 1, layer_size}
            t_w = t_w.reshape({kLstmGates, config.lstm_size, config.lstm_size}).transpose(1, 2);
            t_u = t_u.reshape({kLstmGates, config.lstm_size, config.lstm_size}).transpose(1, 2);
            t_b = t_b.reshape({kLstmGates, 1, config.lstm_size});
            // For non-obvious reasons the LSTM kernel runs faster if the U and W (or _ih) matrices are
            // spaced such that there is room for one more matrix between them. t_w used twice does that.
            t_w = torch::concat({t_u, t_w, t_w, t_b}, 1);

            // reorder from IFGO to GIFO (2, 0, 1, 3), and transpose to gate last
            t_w = torch::stack({t_w[2], t_w[0], t_w[1], t_w[3]}, 2);

            rnn->t_weights_bias.view_as(t_w) = t_w;
        }

        // Load and prepare linear layer weights.
        auto get_linear_weights = [](MetalLinear &linear,
                                     bool use_bias) -> NS::SharedPtr<MTL::Buffer> {
            auto params = linear->named_parameters();
            auto t_w = *params.find("weight");
            const auto num_states = t_w.size(0);
            auto t_b = use_bias ? *params.find("bias") : torch::zeros({num_states});
            if (!use_bias) {
                assert(!params.find("bias"));
            }
            auto linear_w = torch::concat({t_w.transpose(0, 1), t_b.unsqueeze(0)}, 0)
                                    .contiguous()
                                    .to(torch_dtype);
            return extract_mtl_from_tensor(std::move(linear_w));
        };
        if (config.out_features.has_value()) {
            // Linear layer is decomposed into 2 matmuls. the first with a bias.
            linear_weights[0] = get_linear_weights(linear1, true);
            linear_weights[1] = get_linear_weights(linear2, false);
        } else {
            linear_weights[0] = get_linear_weights(linear1, config.bias);
        }
    }

    // Executes the model, with the linear layer held off by linear_hold_off, if non-NULL.
    // If CB submissions are successful, it returns the command buffer used for the linear layer
    // and scan kernels.  If either CB is unsuccessful, it returns nullptr.
    MTL::CommandBuffer *forward_async(at::Tensor &in,
                                      MTL::SharedEvent *const linear_hold_off_event,
                                      uint64_t linear_hold_off_id,
                                      int try_count,
                                      std::vector<at::Tensor> &out) {
        auto command_buffer = m_command_queue->commandBuffer();

        if (in.dtype() != torch::kF16) {
            throw std::runtime_error("Input tensor must be float16.");
        }
        conv1->run(command_buffer, mtl_for_tensor(in), mat_working_mem.get());
        conv2->run(command_buffer, mat_working_mem.get(), mat_temp.get());
        conv3->run(command_buffer, mat_temp.get(), mat_working_mem.get());
        if (!finishCommandBuffer("convolutions", command_buffer, try_count)) {
            return nullptr;
        }

        std::string lstm_label = "lstm_rnn0";
        for (auto &rnn : {rnn1, rnn2, rnn3, rnn4, rnn5}) {
            lstm_label.back()++;
            command_buffer = m_command_queue->commandBuffer();

            const int kResBufSize =
                    static_cast<int>(dtype_bytes * kernel_simd_groups * 2 * kTileSize * kTileSize);
            const int kOutBufSize =
                    static_cast<int>(dtype_bytes * kernel_simd_groups * kTileSize * kTileSize);
            const std::vector<int> tg_buffer_lens{kResBufSize, kOutBufSize};
            for (const auto &args_lstm : m_args_lstm) {
                const std::vector<MTL::Buffer *> buffers{args_lstm.get(), mat_working_mem.get(),
                                                         mtl_for_tensor(rnn->t_weights_bias),
                                                         mat_state.get()};
                launch_kernel_no_wait(lstm_cps[rnn->reverse].get(), command_buffer, buffers,
                                      tg_buffer_lens, kernel_thread_groups,
                                      kernel_simd_groups * kSIMDGroupWidth);
            }

            if (!finishCommandBuffer(lstm_label, command_buffer, try_count)) {
                return nullptr;
            }
        }

        command_buffer = m_command_queue->commandBuffer();

        // The output buffers of conv/LSTM layers are not used by the decoding, so
        // can be overwritten by subsequent batches as soon as they have been consumed by
        // the linear layer.  The output of the linear layer must be protected until
        // it has been decoded.
        if (linear_hold_off_event) {
            command_buffer->encodeWait(linear_hold_off_event, linear_hold_off_id);
        }

        // For now the same SIMD group count, and therefore threadgroup memory buffer size, is
        // used for all linear layer kernel invocations.
        const int kLinearTGOutBufSize =
                static_cast<int>(dtype_bytes * kernel_simd_groups * kTileSize * kTileSize);
        const std::vector<int> linear_tg_buffer_lens{kLinearTGOutBufSize};

        // The output of the linear layer is split into multiple buffers, each generated
        // by a separate kernel launch.
        for (size_t i = 0; i < out.size(); ++i) {
            MTL::Buffer *const args_buffer = args_linear.at(i).get();
            MTL::Buffer *const out_buffer = mtl_for_tensor(out.at(i));
            if (config.out_features.has_value()) {
                launch_kernel_no_wait(linear_cps[0].get(), command_buffer,
                                      {args_buffer, mat_working_mem.get(), linear_weights[0].get(),
                                       mat_temp.get()},
                                      linear_tg_buffer_lens, kernel_thread_groups,
                                      kernel_simd_groups * kSIMDGroupWidth);
                launch_kernel_no_wait(
                        linear_cps[1].get(), command_buffer,
                        {args_linear2.get(), mat_temp.get(), linear_weights[1].get(), out_buffer},
                        linear_tg_buffer_lens, kernel_thread_groups,
                        kernel_simd_groups * kSIMDGroupWidth);
            } else {
                launch_kernel_no_wait(
                        linear_cps[0].get(), command_buffer,
                        {args_buffer, mat_working_mem.get(), linear_weights[0].get(), out_buffer},
                        linear_tg_buffer_lens, kernel_thread_groups,
                        kernel_simd_groups * kSIMDGroupWidth);
            }
        }
        return command_buffer;
    }

    MTL::Device *m_device;
    NS::SharedPtr<MTL::CommandQueue> m_command_queue;
    NS::SharedPtr<MTL::ComputePipelineState> lstm_cps[2], linear_cps[2];
    NS::SharedPtr<MTL::Buffer> mat_working_mem, mat_state, mat_temp, linear_weights[2],
            args_linear2;
    // Each args buffer corresponds to a different time span of the LSTM layer.
    std::vector<NS::SharedPtr<MTL::Buffer>> m_args_lstm;
    std::vector<NS::SharedPtr<MTL::Buffer>> args_linear;
    int in_chunk_size, lstm_chunk_size, batch_size, kernel_thread_groups, kernel_simd_groups;
    CRFModelConfig config;
    MetalLSTM rnn1{nullptr}, rnn2{nullptr}, rnn3{nullptr}, rnn4{nullptr}, rnn5{nullptr};
    MetalConv1d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    MetalLinear linear1{nullptr}, linear2{nullptr};
};

TORCH_MODULE(MetalBlock);

struct MetalModelImpl : Module {
    MetalModelImpl(const CRFModelConfig &config,
                   int chunk_size,
                   int batch_size,
                   int out_split,
                   MTL::Device *const device) {
        mtl_block = register_module("mtl_block",
                                    MetalBlock(chunk_size, batch_size, config, out_split, device));
    }

    void load_state_dict(const std::vector<at::Tensor> &weights) {
        utils::load_state_dict(*this, weights, {});
        mtl_block->load_weights();
    }

    MTL::CommandBuffer *forward_async(at::Tensor &in,
                                      MTL::SharedEvent *const linear_hold_off_event,
                                      uint64_t linear_hold_off_id,
                                      int try_count,
                                      std::vector<at::Tensor> &out) {
        return mtl_block->forward_async(in, linear_hold_off_event, linear_hold_off_id, try_count,
                                        out);
    }

    MetalBlock mtl_block{nullptr};
};

TORCH_MODULE(MetalModel);

}  // namespace nn

class MetalCaller {
    static constexpr int MTL_CORE_BATCH_SIZE = 48;

public:
    MetalCaller(const CRFModelConfig &model_config, int chunk_size, int batch_size)
            : m_config(model_config) {
        ScopedAutoReleasePool autorelease_pool;

        m_num_input_features = model_config.num_features;

        m_device = get_mtl_device();

        m_decoder_options = decode::DecoderOptions();
        m_decoder_options.q_shift = model_config.qbias;
        m_decoder_options.q_scale = model_config.qscale;

        // TODO -- we don't honour the config n_base
        constexpr int n_base = 4;
        m_states = pow(n_base, model_config.state_len);

        // v3 scores come from a tanh activation whose [-1, 1] range is packed into bytes.
        // The linear kernel scales to [-127, 127] byte range, after which beam search
        // rescales to the expected [-5, 5].
        // v4 scores come from a clamped [-5, 5] range that is rescaled by the kernel to
        // fit into bytes.
        // In both cases beam search applies the same 5/127 factor to scores.
        score_scale = static_cast<float>(5.0 / 127.0);

        auto state_dict = load_crf_model_weights(
                model_config.model_path, model_config.out_features.has_value(), model_config.bias);

        if (batch_size == 0) {
            const size_t physical_memory = get_apple_physical_memory_bytes();
            spdlog::debug("Physical memory available {} GB", physical_memory / (size_t{1} << 30));

            // Constrain the maximum batch size to use about half physical memory for decode buffers,
            // with neural network GPU buffers and CPU buffers assumed to occupy a subset of the
            // remaining memory.  This generally constrains the batch size to use fewer than
            // the maximum GPU cores when running sup models on systems with a large GPU core
            // to system memory ratio.
            const auto out_chunk_size = static_cast<size_t>(chunk_size / model_config.stride);
            const auto decode_buffer_size_per_elem =
                    static_cast<size_t>(out_chunk_size) *
                    (static_cast<size_t>(model_config.outsize) +        // Scores
                     static_cast<size_t>(m_states) * sizeof(int16_t) +  // Posts
                     static_cast<size_t>(m_states) * sizeof(float));    // Back guides.
            spdlog::debug("decode_buffer_size_per_elem {}", decode_buffer_size_per_elem);
            const int max_batch_size = static_cast<int>(std::clamp(
                    utils::pad_to(physical_memory / (2 * decode_buffer_size_per_elem),
                                  static_cast<size_t>(MTL_CORE_BATCH_SIZE)),
                    static_cast<size_t>(MTL_CORE_BATCH_SIZE),
                    static_cast<size_t>(MTL_CORE_BATCH_SIZE * get_mtl_device_core_count())));
            spdlog::debug("max_batch_size {}", max_batch_size);

            // Subject to the above memory constraint, impose a minimum batch size
            // that will use 1/4 of GPU cores for LSTM execution.
            const int min_batch_size =
                    std::min(MTL_CORE_BATCH_SIZE * get_mtl_device_core_count() / 4, max_batch_size);
            spdlog::debug("min_batch_size {}", min_batch_size);

            std::set<int> test_batch_sizes{max_batch_size};

            // Add some batch sizes evenly distributed in between.
            const int kNumSmallerSizes = 16;
            const float test_size_increment = static_cast<float>(max_batch_size - min_batch_size) /
                                              static_cast<float>(kNumSmallerSizes);
            for (int i = 0; i < kNumSmallerSizes; ++i) {
                const int test_batch_size =
                        utils::pad_to(min_batch_size + static_cast<int>(i * test_size_increment),
                                      static_cast<int>(MTL_CORE_BATCH_SIZE));
                test_batch_sizes.insert(test_batch_size);
            }

            // To speed up test runs, use a smaller chunk size.  This means we will not see
            // the true effect of memory thrashing, so we are relying on the memory limit
            // above to avoid that scenario.
            const int benchmark_chunk_size = std::min(chunk_size - chunk_size % model_config.stride,
                                                      model_config.stride * 300);

            // Iterate through batch size candidates to find the most efficient one.
            int best_batch_size = -1;
            long long best_us_per_batch_element = std::numeric_limits<long long>::max();
            for (int batch_size : test_batch_sizes) {
                spdlog::debug("Trying batch size {}", batch_size);
                set_chunk_batch_size(model_config, state_dict, benchmark_chunk_size, batch_size);
                auto dummy_input = torch::empty(
                        {batch_size, benchmark_chunk_size, m_num_input_features}, torch::kF16);
                const auto start_time = std::chrono::system_clock::now();
                auto *cb = m_model->forward_async(dummy_input, nullptr, 0, 0, m_scores_int8);
                run_scan_kernels(cb, 0);
                const auto end_time = std::chrono::system_clock::now();
                const auto elapsed_us =
                        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)
                                .count();
                const auto us_per_batch_element = elapsed_us / batch_size;
                spdlog::debug("Batch {} us Batch element {} us", elapsed_us, us_per_batch_element);
                if (us_per_batch_element < best_us_per_batch_element) {
                    best_us_per_batch_element = us_per_batch_element;
                    best_batch_size = batch_size;
                }
            }
            assert(best_batch_size >= MTL_CORE_BATCH_SIZE);
            assert(best_batch_size % MTL_CORE_BATCH_SIZE == 0);
            set_chunk_batch_size(model_config, state_dict, chunk_size, best_batch_size);
        } else {
            // Use the user-supplied batch size padded to the nearest reasonable value.
            set_chunk_batch_size(model_config, state_dict, chunk_size,
                                 utils::pad_to(batch_size, MTL_CORE_BATCH_SIZE));
        }

        start_threads();
    }

    void set_chunk_batch_size(const CRFModelConfig &model_config,
                              const std::vector<at::Tensor> &state_dict,
                              int chunk_size,
                              int batch_size) {
        // Chunk size after decimation via convolution stride.
        m_out_chunk_size = chunk_size / model_config.stride;
        // round chunk size down to a multiple of the stride
        m_in_chunk_size = m_out_chunk_size * model_config.stride;

        m_batch_size = batch_size;

        // Allocations beyond 4GB can fail, and the linear layer output buffer
        // hits this limit with batch sizes larger than 384 with typical
        // chunk sizes.  We also want to limit memory usage in general.
        // At the same time, the LSTM layer performance benefits
        // from large batch sizes.
        // We therefore run the linear layer via 1 or more kernel runs, each
        // with an output buffer of limited size, aiming for <= kMaxBufferSize.
        // As things stand, we need an exactly even split of batch elements in
        // the linear layer output buffers (this could be relaxed).
        // We therefore want the smallest divisor of batch_size that results in
        // linear layer output buffers <= kMaxBufferSize, and a linear layer batch size
        // that is an integral multiple of 48.  Since the LSTM batch size is
        // already constrained to be an integral multiple of 48, this means the
        // batch splitting factor must be an exact divisor of the batch_size / 48.

        // If this target is smaller than the size required for 48 batch elements, then
        // that size is the best we can do.  The size here is attainable for fast and hac
        // models, but not sup.
        constexpr auto kMaxBufferSize = static_cast<int64_t>(1) << 29;
        const auto complete_linear_out_size =
                static_cast<int64_t>(m_out_chunk_size) * static_cast<int64_t>(m_batch_size) *
                static_cast<int64_t>(model_config.outsize) * sizeof(float);
        const int num_batch_pieces = m_batch_size / MTL_CORE_BATCH_SIZE;
        for (m_out_split = 1; m_out_split < num_batch_pieces; ++m_out_split) {
            if (num_batch_pieces % m_out_split == 0 &&
                complete_linear_out_size / m_out_split <= kMaxBufferSize)
                break;
        }
        auto piece_size = complete_linear_out_size / m_out_split;
        if (piece_size > kMaxBufferSize) {
            spdlog::debug("Did not hit linear layer target output size {} - got {}", kMaxBufferSize,
                          piece_size);
        }
        spdlog::debug("Linear layer split {}", m_out_split);
        // If we exited the loop above without breaking, then m_out_split = num_batch_pieces,
        // which satisfies the divisor criterion, and should mean small enough linear layer
        // output buffers, given other reasonable parameters.
        assert(num_batch_pieces % m_out_split == 0);
        assert(m_batch_size % m_out_split == 0);
        m_out_batch_size = m_batch_size / m_out_split;
        assert(m_out_batch_size % MTL_CORE_BATCH_SIZE == 0);

        m_model = nn::MetalModel(model_config, m_in_chunk_size, m_batch_size, m_out_split,
                                 m_device.get());
        m_model->load_state_dict(state_dict);
        m_model->eval();

        m_decode_complete_event = NS::TransferPtr(m_device->newSharedEvent());
        m_bwd_scan_cps = make_cps(m_device.get(), "backward_scan", {}, std::nullopt);
        m_fwd_scan_add_softmax_cps =
                make_cps(m_device.get(), "forward_scan_add_softmax", {}, std::nullopt);

        int T = m_out_chunk_size;
        int C = model_config.outsize;
        int Cs = m_states;

        m_scores_int8.clear();
        m_posts_int16.clear();
        m_bwd.clear();
        for (int i = 0; i < m_out_split; ++i) {
            m_scores_int8.push_back(torch::empty({T, m_out_batch_size, C}, torch::kInt8));
            // Unfortunately torch doesn't have Uint16, or we would use it.  We could offset,
            // or rely on undefined overflow behaviour, but for now we waste the sign bit.
            m_posts_int16.push_back(torch::empty({m_out_batch_size, T + 1, Cs}, torch::kInt16));
            m_bwd.push_back(torch::empty({m_out_batch_size, T + 1, Cs}));
        }
    }

    void start_threads() {
        m_metal_thread.reset(new std::thread(&MetalCaller::metal_thread_fn, this));

        int num_decode_threads = std::max(1, get_apple_cpu_perf_core_count() - 1);
        m_decode_threads.reserve(num_decode_threads);
        for (int i = 0; i < num_decode_threads; ++i) {
            m_decode_threads.emplace_back(new std::thread(&MetalCaller::decode_thread_fn, this));
        }
    }

    ~MetalCaller() {
        m_terminate.store(true);
        m_input_cv.notify_one();
        m_decode_cv.notify_all();

        if (m_metal_thread && m_metal_thread->joinable()) {
            m_metal_thread->join();
        }
        for (auto &thr : m_decode_threads) {
            thr->join();
        }
    }

    struct NNTask {
        NNTask(at::Tensor *input_, int num_chunks_, std::vector<decode::DecodedChunk> *out_chunks_)
                : input(input_), out_chunks(out_chunks_), num_chunks(num_chunks_) {}

        at::Tensor *input;
        std::mutex mut;
        std::condition_variable cv;
        bool ready{false};
        std::vector<decode::DecodedChunk> *out_chunks;
        int num_chunks;
        int decode_chunks_started{0};
        int decode_chunks_finished{0};
        // Event ID to be signalled when decoding for this task is complete, set by metal_thread_fn.
        uint64_t decode_complete_event_id = static_cast<uint64_t>(0);
    };

    void call_chunks(at::Tensor &input,
                     int num_chunks,
                     std::vector<decode::DecodedChunk> &out_chunks) {
        if (num_chunks == 0) {
            return;
        }

        auto task = std::make_shared<NNTask>(&input, num_chunks, &out_chunks);
        {
            std::lock_guard<std::mutex> lock(m_input_lock);
            m_input_queue.push_front(task);
        }
        m_input_cv.notify_one();

        std::unique_lock lock(task->mut);
        while (task->decode_chunks_finished != num_chunks) {
            task->cv.wait(lock);
        }
    }

    bool run_scan_kernels(MTL::CommandBuffer *const cb, int try_count) {
        // This stage is operating on the split outputs of the linear layer, so
        // the effective batch size is m_out_batch_size.
        std::vector<int32_t> scan_args_{m_out_chunk_size, m_out_batch_size, m_states};
        auto scan_args = create_vec_buffer(m_device.get(), scan_args_);

        for (int i = 0; i < m_out_split; ++i) {
            // TODO: optimise grid size
            launch_kernel_no_wait(m_bwd_scan_cps.get(), cb,
                                  {scan_args.get(), mtl_for_tensor(m_scores_int8.at(i)),
                                   mtl_for_tensor(m_bwd.at(i))},
                                  {}, m_out_batch_size, m_states);

            launch_kernel_no_wait(
                    m_fwd_scan_add_softmax_cps.get(), cb,
                    {scan_args.get(), mtl_for_tensor(m_scores_int8.at(i)),
                     mtl_for_tensor(m_bwd.at(i)), mtl_for_tensor(m_posts_int16.at(i))},
                    {}, m_out_batch_size, m_states);
        }
        return finishCommandBuffer("linear/scan/softmax", cb, try_count);
    }

    void metal_thread_fn() {
        at::InferenceMode inference_mode_guard;
        ScopedAutoReleasePool autorelease_pool;

        // Incrementing ID used to prevent the linear layer of run i+1 overwriting the scores of
        // run i before the CPU has finished decoding all run i's chunks.
        // Start at 1, since at event creation ID 0 is deemed to have been signalled.
        auto next_decode_complete_event_id = static_cast<uint64_t>(1);

        // For unknown reasons, concurrent access to the GPU from multiple instances of this thread --
        // i.e. with > 1 instance of MetalCaller -- results in errors, usually command buffer error code 1.
        // Holding this mutex while executing models seemingly prevents these errors.
        static std::mutex inter_caller_mutex;

        while (true) {
            std::unique_lock<std::mutex> input_lock(m_input_lock);
            while (m_input_queue.empty() && !m_terminate.load()) {
                m_input_cv.wait_for(input_lock, 100ms);
            }

            if (m_input_queue.empty() && m_terminate.load()) {
                m_terminate_decode.store(true);
                return;
            }

            auto task = std::move(m_input_queue.back());
            m_input_queue.pop_back();
            input_lock.unlock();

            // Assign this task a unique decode completion event ID.
            // This ID will be signalled by the CPU once it has finished relevant decoding work,
            // allowing the GPU to proceed.
            task->decode_complete_event_id = next_decode_complete_event_id++;

            // We retry the entire set of kernels up to 5 times, to deal with seemingly
            // random intermittent errors with command buffer submissions.
            // TODO: find a more robust way of dealing with Metal kernel launch issues
            bool cb_success = false;
            for (int try_count = 0; try_count < 5; ++try_count) {
                std::lock_guard<std::mutex> lock(inter_caller_mutex);

                // The linear layer should not execute until the previous batch has been decoded,
                // since the same buffers are used for successive batches' scores, fwd/bwd scans.
                MTL::CommandBuffer *const cb = m_model->forward_async(
                        *task->input, m_decode_complete_event.get(),
                        task->decode_complete_event_id - 1, try_count, m_scores_int8);
                if (cb == nullptr) {
                    // A command buffer submission part of forward_async failed, so we should retry.
                    std::this_thread::sleep_for(20ms);
                    continue;
                }

                if (run_scan_kernels(cb, try_count)) {
                    cb_success = true;
                    break;
                }

                // linear/scan/softmax CB failed, so retry.
                std::this_thread::sleep_for(20ms);
            }

            // If we repeatedly submitted CBs without success, we give up.
            if (!cb_success) {
                spdlog::critical("Failed to successfully submit GPU command buffers.");
                throw std::runtime_error("Failed to successfully submit GPU command buffers.");
            }

            // Pass task on to decode threads
            std::unique_lock<std::mutex> decode_lock(m_decode_lock);
            m_decode_queue.push_front(task);
            decode_lock.unlock();
            m_decode_cv.notify_all();
        }
    }

    void decode_thread_fn() {
        at::InferenceMode inference_mode_guard;
        while (true) {
            std::unique_lock<std::mutex> decode_lock(m_decode_lock);
            while (m_decode_queue.empty() && !m_terminate_decode.load()) {
                m_decode_cv.wait_for(decode_lock, 100ms);
            }

            if (m_decode_queue.empty() && m_terminate_decode.load()) {
                return;
            }
            auto task = m_decode_queue.back();
            int chunk_idx = task->decode_chunks_started++;
            // If all chunks have been picked up for decoding, remove task from queue
            if (chunk_idx == task->num_chunks - 1) {
                m_decode_queue.pop_back();
            }
            decode_lock.unlock();

            // Model outputs are split across m_out_split buffers.
            assert(m_scores_int8.size() == static_cast<size_t>(m_out_split));
            assert(m_bwd.size() == static_cast<size_t>(m_out_split));
            assert(m_posts_int16.size() == static_cast<size_t>(m_out_split));
            const int out_buf_idx = chunk_idx / m_out_batch_size;
            const int buf_chunk_idx = chunk_idx % m_out_batch_size;

            auto [sequence, qstring, moves] = decode::beam_search_decode(
                    m_scores_int8.at(out_buf_idx).index({Slice(), buf_chunk_idx}),
                    m_bwd.at(out_buf_idx)[buf_chunk_idx],
                    m_posts_int16.at(out_buf_idx)[buf_chunk_idx], m_decoder_options.beam_width,
                    m_decoder_options.beam_cut, m_decoder_options.blank_score,
                    m_decoder_options.q_shift, m_decoder_options.q_scale, score_scale);

            (*task->out_chunks)[chunk_idx] =
                    decode::DecodedChunk{std::move(sequence), std::move(qstring), std::move(moves)};

            // Wake the waiting thread which called `call_chunks()` if we're done decoding
            std::unique_lock<std::mutex> task_lock(task->mut);
            bool done = ++(task->decode_chunks_finished) == task->num_chunks;
            task_lock.unlock();
            if (done) {
                // Now that all chunks are decoded, signal that the GPU can overwrite the scores
                // buffer with subsequent work.
                assert(m_decode_complete_event);
                m_decode_complete_event->setSignaledValue(task->decode_complete_event_id);
                task->cv.notify_one();
            }
        }
    }

    void terminate() {
        m_terminate.store(true);
        m_input_cv.notify_one();
        m_decode_cv.notify_all();
        if (m_metal_thread && m_metal_thread->joinable()) {
            m_metal_thread->join();
        }
        m_metal_thread.reset();
        for (auto &thr : m_decode_threads) {
            if (thr->joinable()) {
                thr->join();
            }
        }
        m_decode_threads.clear();
    }

    void restart() {
        // This can be called more than one, via multiple runners.
        if (m_terminate.load()) {
            m_terminate.store(false);
            m_terminate_decode.store(false);
            start_threads();
        }
    }

    const CRFModelConfig m_config;
    std::atomic<bool> m_terminate{false};
    std::atomic<bool> m_terminate_decode{false};
    std::deque<std::shared_ptr<NNTask>> m_input_queue;
    std::deque<std::shared_ptr<NNTask>> m_decode_queue;
    std::mutex m_input_lock;
    std::condition_variable m_input_cv;
    std::unique_ptr<std::thread> m_metal_thread;
    std::mutex m_decode_lock;
    std::condition_variable m_decode_cv;
    std::vector<std::unique_ptr<std::thread>> m_decode_threads;
    decode::DecoderOptions m_decoder_options;
    nn::MetalModel m_model{nullptr};
    NS::SharedPtr<MTL::Device> m_device;
    NS::SharedPtr<MTL::ComputePipelineState> m_bwd_scan_cps, m_fwd_scan_add_softmax_cps;
    // Used to signal completion of an NNTask's decoding.
    NS::SharedPtr<MTL::SharedEvent> m_decode_complete_event;
    std::vector<at::Tensor> m_scores_int8, m_posts_int16, m_bwd;
    int m_in_chunk_size, m_out_chunk_size, m_batch_size, m_states;
    // Number of pieces the linear output is split into, for reasons of
    // buffer size constraints.
    int m_out_split;
    int m_out_batch_size;
    // v3 and v4 models have different score scaling requirements.
    float score_scale{0.0f};
    // Chunk input channel count.
    int m_num_input_features = -1;
};

std::shared_ptr<MetalCaller> create_metal_caller(const CRFModelConfig &model_config,
                                                 int chunk_size,
                                                 int batch_size) {
    return std::make_shared<MetalCaller>(model_config, chunk_size, batch_size);
}

MetalModelRunner::MetalModelRunner(std::shared_ptr<MetalCaller> caller) : m_caller(caller) {
    // Metal convolution kernels operate with channel ordering (N, T, C).  If m_input
    // is to be submitted directly then it must also have this arrangement.
    // Note that this is not the same as other caller implementations, which
    // have T innermost.
    m_input = torch::empty(
            {caller->m_batch_size, caller->m_in_chunk_size, caller->m_num_input_features},
            torch::kF16);
}

void MetalModelRunner::accept_chunk(int chunk_idx, const at::Tensor &chunk) {
    if (chunk.dim() == 1) {
        // Input has single feature dimension.
        assert(m_caller->m_num_input_features == 1);
        m_input.index_put_({chunk_idx, Ellipsis, 0}, chunk);
    } else {
        // Chunks are passed with timestep the innermost dimension, whereas we need
        // channels innermost.
        assert(m_caller->m_num_input_features == chunk.size(0));
        m_input.index_put_({chunk_idx, Ellipsis, Ellipsis}, chunk.transpose(0, 1));
    }
}

std::vector<decode::DecodedChunk> MetalModelRunner::call_chunks(int num_chunks) {
    ++m_num_batches_called;
    std::vector<decode::DecodedChunk> out_chunks(num_chunks);
    m_caller->call_chunks(m_input, num_chunks, out_chunks);
    return out_chunks;
}

const CRFModelConfig &MetalModelRunner::config() const { return m_caller->m_config; }
size_t MetalModelRunner::model_stride() const { return m_caller->m_config.stride; }
size_t MetalModelRunner::chunk_size() const { return m_input.size(1); }
size_t MetalModelRunner::batch_size() const { return m_input.size(0); }

void MetalModelRunner::terminate() { m_caller->terminate(); }
void MetalModelRunner::restart() { m_caller->restart(); }

stats::NamedStats MetalModelRunner::sample_stats() const {
    stats::NamedStats stats;
    stats["batches_called"] = m_num_batches_called;
    return stats;
}

}  // namespace dorado::basecall
