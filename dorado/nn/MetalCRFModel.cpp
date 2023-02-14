#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "MetalCRFModel.h"

#include "../decode/beam_search.h"
#include "../utils/metal_utils.h"
#include "../utils/module_utils.h"
#include "../utils/tensor_utils.h"

#include <math.h>
#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <vector>

using namespace dorado::utils;
using namespace std::chrono_literals;
using namespace torch::nn;
namespace F = torch::nn::functional;
using Slice = torch::indexing::Slice;

static constexpr auto torch_dtype = torch::kF16;
static const size_t dtype_bytes = torch::elementSize(torch_dtype);

namespace {
// SIMD tile size dictated by the metal spec.
const int kTileSize = 8;

bool finishCommandBuffer(const char *label, MTL::CommandBuffer *cb, int try_count) {
    cb->commit();
    cb->waitUntilCompleted();

    auto status = cb->status();
    bool success = (status == MTL::CommandBufferStatusCompleted);
    if (success) {
        spdlog::debug("Metal command buffer {}: {} ms", label,
                      1000.f * float(cb->GPUEndTime() - cb->GPUStartTime()));
    } else {
        spdlog::warn("Metal command buffer {} failed: {}, try #{}", label, status, try_count);
        if (status == MTL::CommandBufferStatusError) {
            const auto *const error_ptr = cb->error();
            if (error_ptr)
                spdlog::warn("Command buffer error code: {}", error_ptr->code());
        }
    }
    return success;
}

}  // namespace

namespace dorado {

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
                    bool clamp,
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
            assert(out_size == 4 || out_size == 16);
            assert(win_size = 5);
        } else if (layer == 2) {
            assert(in_size == 4 || in_size == 16);
            assert(out_size == 16);
            assert(win_size = 5);
        }

        const std::vector<int32_t> args_{in_size,      win_size,   out_size,  stride,
                                         win_size / 2, chunk_size, batch_size};
        args = create_vec_buffer(device, args_);

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
                {"kConvOutputClamp", clamp}};
        const int kernel_threads = 32 * kernel_simd_groups;
        std::string kernel_name = "conv" + std::to_string(layer);
        // Layers 1 and 2 have variants with a different intermediate feature size.
        if (layer == 1)
            kernel_name += "_out" + std::to_string(out_size);
        else if (layer == 2)
            kernel_name += "_in" + std::to_string(in_size);
        conv_cps = make_cps(device, kernel_name + "_simd", metal_constants, kernel_threads);
    }

    void run(MTL::CommandQueue *command_queue, MTL::Buffer *mat_in, MTL::Buffer *mat_out) {
        std::vector<MTL::Buffer *> buffers{args, mat_in, mtl_for_tensor(t_weights_bias), mat_out};
        launch_kernel(conv_cps, command_queue, buffers, {}, kernel_thread_groups,
                      kernel_simd_groups * 32);
    }

    void run(MTL::CommandBuffer *command_buffer, MTL::Buffer *mat_in, MTL::Buffer *mat_out) {
        std::vector<MTL::Buffer *> buffers{args, mat_in, mtl_for_tensor(t_weights_bias), mat_out};
        launch_kernel_no_wait(conv_cps, command_buffer, buffers, {}, kernel_thread_groups,
                              kernel_simd_groups * 32);
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

    torch::Tensor t_weights_bias;
    MTL::Buffer *args;
    MTL::ComputePipelineState *conv_cps, *weights_cps;
    int kernel_simd_groups, kernel_thread_groups;
    int in_size, out_size, win_size, stride, chunk_size, batch_size, w_pad_rows, repeats;
};

TORCH_MODULE(MetalConv1d);

static constexpr int kLstmGates = 4;
struct MetalLSTMImpl : Module {
    MetalLSTMImpl(int layer_size, bool reverse_, MTL::Device *device) : reverse(reverse_) {
        auto weight_ih = torch::empty({layer_size * kLstmGates, layer_size});
        auto weight_hh = torch::empty({layer_size * kLstmGates, layer_size});
        auto bias_ih = torch::empty({layer_size * kLstmGates});
        auto bias_hh = torch::empty({layer_size * kLstmGates});

        register_parameter("weight_ih", weight_ih, false);
        register_parameter("weight_hh", weight_hh, false);
        register_parameter("bias_ih", bias_ih, false);
        register_parameter("bias_hh", bias_hh, false);

        t_weights_bias = torch::empty({layer_size * 2 + 1, layer_size, kLstmGates}, torch_dtype);
    }

    torch::Tensor t_weights_bias;
    bool reverse;
};

TORCH_MODULE(MetalLSTM);

struct MetalBlockImpl : Module {
    MetalBlockImpl(int chunk_size_,
                   int batch_size_,
                   const CRFModelConfig &config,
                   int out_split_,
                   MTL::Device *const device_)
            : device(device_),
              in_chunk_size(chunk_size_),
              stride(config.stride),
              batch_size(batch_size_),
              layer_size(config.insize),
              out_size(config.outsize),
              conv(config.conv),
              out_features(config.out_features),
              stay_score(config.blank_score) {
        command_queue = device->newCommandQueue();

        lstm_chunk_size = in_chunk_size / stride;

        // args for LSTM kernel
        {
            std::vector<int32_t> args_lstm_{batch_size / kTileSize, lstm_chunk_size};
            args_lstm = create_vec_buffer(device, args_lstm_);
        }

        // args for conversion to half
        {
            const std::vector<int32_t> args_to_half_{in_chunk_size * batch_size};
            args_to_half = create_vec_buffer(device, args_to_half_);
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
            args_linear.at(i) = create_vec_buffer(device, args_linear_);
        }
        args_linear2 = create_vec_buffer<int32_t>(
                device, {out_batch_tiles, 0, out_batch_tiles, lstm_chunk_size});

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
        const int lstm_threads = kernel_simd_groups * 32;
        lstm_cps[0] = make_cps(device, "lstm",
                               {{"kLstmLayerSize", layer_size}, {"kLstmReversedInTime", false}},
                               lstm_threads);
        lstm_cps[1] = make_cps(device, "lstm",
                               {{"kLstmLayerSize", layer_size}, {"kLstmReversedInTime", true}},
                               lstm_threads);
        to_half_cps = make_cps(device, "float_to_half", {});

        // The temp buffer used for these purposes (number of elements of `torch_dtype` in []):
        // - Store inputs converted to F16 (if torch_dtype == kF16) [in_chunk_size * batch_size]
        // - Store output of second conv layer [in_chunk_size * batch_size * kMaxConv2OutChannels]
        // - Store temp output of lstm layers [batch_size * layer_size]
        // - Store output of first linear layer if there are two
        //   [lstm_chunk_size * batch_size * decomposition / out_split_]
        constexpr int kMaxConv2OutChannels = 16;
        int mat_temp_elems =
                batch_size * std::max(kMaxConv2OutChannels * in_chunk_size, layer_size);

        conv1 = register_module("conv1", MetalConv1d(1, 1, conv, 5, 1, config.clamp, in_chunk_size,
                                                     batch_size, device));
        conv2 = register_module("conv2", MetalConv1d(2, conv, 16, 5, 1, config.clamp, in_chunk_size,
                                                     batch_size, device));
        conv3 = register_module("conv3", MetalConv1d(3, 16, layer_size, 19, stride, config.clamp,
                                                     in_chunk_size, batch_size, device));
        rnn1 = register_module("rnn_1", MetalLSTM(layer_size, true, device));
        rnn2 = register_module("rnn_2", MetalLSTM(layer_size, false, device));
        rnn3 = register_module("rnn_3", MetalLSTM(layer_size, true, device));
        rnn4 = register_module("rnn_4", MetalLSTM(layer_size, false, device));
        rnn5 = register_module("rnn_5", MetalLSTM(layer_size, true, device));

        const int linear_threads = kernel_simd_groups * 32;
        // If the intermediate feature size between conv1 and conv2 is 16, then this is a v4
        // type model, where the linear layer output is clamped rather than run through tanh.
        // Otherwise the intermediate feature size is 4.
        assert(config.conv == 4 || config.conv == 16);
        if (config.out_features.has_value()) {
            // The linear layer is decomposed into 2 matmuls.
            const int decomposition = config.out_features.value();
            linear1 =
                    register_module("linear1", MetalLinear(layer_size, decomposition, config.bias));
            const bool kSecondLayerBias = false;
            linear2 = register_module("linear2",
                                      MetalLinear(decomposition, out_size, kSecondLayerBias));
            const auto linear_constants1 = std::vector<std::tuple<std::string, MetalConstant>>(
                    {{"kLinearInSize", layer_size},
                     {"kLinearOutSize", decomposition},
                     {"kLinearOutputScale", 1.0f},
                     {"kLinearOutputClamp", false},
                     {"kLinearOutputTanh", false},
                     {"kLinearOutputAsByte", false}});
            linear_cps[0] = make_cps(device, "linear_from_lstm", linear_constants1, linear_threads);
            const auto linear_constants2 = std::vector<std::tuple<std::string, MetalConstant>>(
                    {{"kLinearInSize", decomposition},
                     {"kLinearOutSize", out_size},
                     // Rescale from clamped [-5.0, 5.0] range to byte range.
                     {"kLinearOutputScale", 127.0f / 5.0f},
                     {"kLinearOutputClamp", true},
                     {"kLinearOutputTanh", false},
                     {"kLinearOutputAsByte", true}});
            linear_cps[1] = make_cps(device, "linear", linear_constants2, linear_threads);
            mat_temp_elems = std::max(mat_temp_elems,
                                      decomposition * (batch_size / out_split_) * lstm_chunk_size);
        } else {
            bool is_v3_model = (config.conv == 4);
            const auto linear_constants = std::vector<std::tuple<std::string, MetalConstant>>(
                    {{"kLinearInSize", layer_size},
                     {"kLinearOutSize", out_size},
                     // If V4, rescale from clamped [-5.0, 5.0] range to byte range.
                     {"kLinearOutputScale", is_v3_model ? 127.0f : (127.0f / 5.0f)},
                     {"kLinearOutputClamp", !is_v3_model},
                     {"kLinearOutputTanh", is_v3_model},
                     {"kLinearOutputAsByte", true}});
            linear_cps[0] = make_cps(device, "linear_from_lstm", linear_constants, linear_threads);
            // Single matmul that may or may not have a bias.
            if (!config.out_features.has_value()) {
                linear1 =
                        register_module("linear1", MetalLinear(layer_size, out_size, config.bias));
            }
        }

        // This buffer is used for several layers of the model.
        mat_working_mem = create_buffer(
                device, size_t(lstm_chunk_size + 2) * batch_size * layer_size * dtype_bytes);
        mat_state = create_buffer(device, batch_size * layer_size * dtype_bytes);
        mat_temp = create_buffer(device, mat_temp_elems * dtype_bytes);
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

            // Reshape and combine matrices into one of size {kLstmGates, 2 * layer_size + 1, layer_size}
            t_w = t_w.reshape({kLstmGates, layer_size, layer_size}).transpose(1, 2);
            t_u = t_u.reshape({kLstmGates, layer_size, layer_size}).transpose(1, 2);
            t_b = t_b.reshape({kLstmGates, 1, layer_size});
            t_w = torch::concat({t_u, t_w, t_b}, 1);

            // reorder from IFGO to GIFO (2, 0, 1, 3), and transpose to gate last
            t_w = torch::stack({t_w[2], t_w[0], t_w[1], t_w[3]}, 2);

            rnn->t_weights_bias.view_as(t_w) = t_w;
        }

        // Load and prepare linear layer weights.
        auto get_linear_weights = [](MetalLinear &linear, bool use_bias) -> MTL::Buffer * {
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
            return extract_mtl_from_tensor(linear_w);
        };
        if (out_features.has_value()) {
            linear_weights[0] = get_linear_weights(linear1, true);
            linear_weights[1] = get_linear_weights(linear2, false);
        } else {
            // v3 single matrix with bias, or v4 single matrix without bias.
            linear_weights[0] = get_linear_weights(linear1, (conv == 4));
        }
    }

    // Executes the model, with the linear layer held off by linear_hold_off, if non-NULL.
    MTL::CommandBuffer *forward_async(torch::Tensor &in,
                                      MTL::SharedEvent *const linear_hold_off_event,
                                      int linear_hold_off_id,
                                      std::vector<torch::Tensor> &out) {
        auto command_buffer = command_queue->commandBuffer();

        if (torch_dtype == torch::kF16) {
            // Convert input activations from float32 to float16.
            launch_kernel_no_wait(to_half_cps, command_buffer,
                                  {args_to_half, mtl_for_tensor(in), mat_temp}, {},
                                  kernel_thread_groups, 256);
            conv1->run(command_buffer, mat_temp, mat_working_mem);
        } else {
            conv1->run(command_buffer, mtl_for_tensor(in), mat_working_mem);
        }
        finishCommandBuffer("conv1", command_buffer, 0);
        command_buffer = command_queue->commandBuffer();
        conv2->run(command_buffer, mat_working_mem, mat_temp);
        finishCommandBuffer("conv2", command_buffer, 0);
        command_buffer = command_queue->commandBuffer();
        conv3->run(command_buffer, mat_temp, mat_working_mem);
        finishCommandBuffer("conv3", command_buffer, 0);
        command_buffer = command_queue->commandBuffer();

        for (auto &rnn : {rnn1, rnn2, rnn3, rnn4, rnn5}) {
            const std::vector<MTL::Buffer *> buffers{args_lstm, mat_working_mem,
                                                     mtl_for_tensor(rnn->t_weights_bias), mat_state,
                                                     mat_temp};
            const int kResBufSize = dtype_bytes * kernel_simd_groups * 2 * kTileSize * kTileSize;
            const int kOutBufSize = dtype_bytes * kernel_simd_groups * kTileSize * kTileSize;
            const std::vector<int> tg_buffer_lens{kResBufSize, kOutBufSize};
            launch_kernel_no_wait(lstm_cps[rnn->reverse], command_buffer, buffers, tg_buffer_lens,
                                  kernel_thread_groups, kernel_simd_groups * 32);
        }
        finishCommandBuffer("lstm", command_buffer, 0);
        command_buffer = command_queue->commandBuffer();

        // The output buffers of conv/LSTM layers are not used by the decoding, so
        // can be overwritten by subsequent batches as soon as they have been consumed by
        // the linear layer.  The output of the linear layer must be protected until
        // it has been decoded.
        if (linear_hold_off_event != nullptr) {
            command_buffer->encodeWait(linear_hold_off_event, linear_hold_off_id);
        }

        // For now the same SIMD group count, and therefore threadgroup memory buffer size, is
        // used for all linear layer kernel invocations.
        const int kLinearTGOutBufSize =
                static_cast<int>(dtype_bytes * kernel_simd_groups * kTileSize * kTileSize);
        const std::vector<int> linear_tg_buffer_lens{kLinearTGOutBufSize};

        // The output of the linear layer is split into multiple buffers, each generated
        // by a separate kernel launch.
        for (int i = 0; i < out.size(); ++i) {
            MTL::Buffer *const args_buffer = args_linear.at(i);
            MTL::Buffer *const out_buffer = mtl_for_tensor(out.at(i));
            if (out_features.has_value()) {
                launch_kernel_no_wait(linear_cps[0], command_buffer,
                                      {args_buffer, mat_working_mem, linear_weights[0], mat_temp},
                                      linear_tg_buffer_lens, kernel_thread_groups,
                                      kernel_simd_groups * 32);
                launch_kernel_no_wait(linear_cps[1], command_buffer,
                                      {args_linear2, mat_temp, linear_weights[1], out_buffer},
                                      linear_tg_buffer_lens, kernel_thread_groups,
                                      kernel_simd_groups * 32);
            } else {
                launch_kernel_no_wait(linear_cps[0], command_buffer,
                                      {args_buffer, mat_working_mem, linear_weights[0], out_buffer},
                                      linear_tg_buffer_lens, kernel_thread_groups,
                                      kernel_simd_groups * 32);
            }
        }
        return command_buffer;
    }

    torch::Tensor forward(torch::Tensor in) {
        std::vector<torch::Tensor> out{torch::empty({lstm_chunk_size, batch_size, out_size})};
        // TODO: find a more robust way of dealing with Metal kernel launch issues
        for (int try_count = 0; try_count < 5; ++try_count) {
            MTL::CommandBuffer *command_buffer = forward_async(in, nullptr, 0, out);
            command_buffer->commit();
            command_buffer->waitUntilCompleted();
            if (command_buffer->status() == MTL::CommandBufferStatusCompleted) {
                break;
            }
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(20ms);
        }
        return out.at(0);
    }

    MTL::Device *device;
    MTL::CommandQueue *command_queue;
    std::string fn[2];
    MTL::ComputePipelineState *lstm_cps[2], *to_half_cps, *linear_cps[2];
    MTL::Buffer *mat_working_mem, *mat_state, *mat_temp, *args_lstm, *args_to_half,
            *linear_weights[2], *args_linear2;
    std::vector<MTL::Buffer *> args_linear;
    int in_chunk_size, lstm_chunk_size, stride, batch_size, layer_size, out_size, conv,
            kernel_thread_groups, kernel_simd_groups;
    std::optional<int> out_features;
    float stay_score;
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

    void load_state_dict(const std::vector<torch::Tensor> &weights) {
        utils::load_state_dict(*this, weights);
        mtl_block->load_weights();
    }

    torch::Tensor forward(torch::Tensor x) { return mtl_block->forward(x); }

    MTL::CommandBuffer *forward_async(torch::Tensor &in,
                                      MTL::SharedEvent *const linear_hold_off_event,
                                      int linear_hold_off_id,
                                      std::vector<torch::Tensor> &out) {
        return mtl_block->forward_async(in, linear_hold_off_event, linear_hold_off_id, out);
    }

    MetalBlock mtl_block{nullptr};
};

TORCH_MODULE(MetalModel);

}  // namespace nn

class MetalCaller {
public:
    MetalCaller(const std::filesystem::path &model_path, int chunk_size, int batch_size) {
        // LSTM kernels, which run with the full supplied batch size, require that the batch size
        // be an integral multiple of 48.
        assert(batch_size % 48 == 0);

        const auto model_config = load_crf_model_config(model_path);
        m_model_stride = static_cast<size_t>(model_config.stride);

        m_device = get_mtl_device();

        m_decoder_options = DecoderOptions();
        m_decoder_options.q_shift = model_config.qbias;
        m_decoder_options.q_scale = model_config.qscale;

        // adjust chunk size to a multiple of the stride
        chunk_size -= chunk_size % model_config.stride;

        // TODO -- we don't honour the config n_base
        constexpr int n_base = 4;
        m_states = pow(n_base, model_config.state_len);

        m_batch_size = batch_size;

        // Chunk size after decimation via convolution stride.
        m_out_chunk_size = chunk_size / model_config.stride;

        auto state_dict = load_crf_model_weights(model_path, model_config.out_features.has_value(),
                                                 model_config.bias);

        // Allocations beyond 4GB can fail, and the linear layer output buffer
        // hits this limit with batch sizes larger than 384 with typical
        // chunk sizes.  At the same time, the LSTM layer performance benefits
        // from large batch sizes.
        // We therefore run the linear layer via 1 or more kernel runs, each
        // with an output buffer with a size <= 4GB, with a reduced batch size.
        // The linear layer kernel requires a batch size that is an integral
        // multiple of 48.
        // As things stand, we need an exactly even split of batch elements in
        // the linear layer output buffers (this could be relaxed).
        // We therefore need the smallest divisor of batch_size that results in
        // linear layer output buffers < 4GB, and a linear layer batch size
        // that is an integral multiple of 48.  Since the LSTM batch size is
        // already constrained to be an integral multiple of 48, this means the
        // batch splitting factor must be an exact divisor of the batch_size / 48.
        constexpr auto kMaxBufferSize = static_cast<int64_t>(1) << 32;
        const auto complete_linear_out_size =
                static_cast<int64_t>(m_out_chunk_size) * static_cast<int64_t>(batch_size) *
                static_cast<int64_t>(model_config.outsize) * sizeof(float);
        const int num_batch_pieces = batch_size / 48;
        for (m_out_split = 1; m_out_split < num_batch_pieces; ++m_out_split) {
            if (num_batch_pieces % m_out_split == 0 &&
                complete_linear_out_size / m_out_split < kMaxBufferSize)
                break;
        }
        // If we exited the loop above without breaking, then m_out_split = num_batch_pieces,
        // which satisfies the divisor criterion, and should mean small enough linear layer
        // output buffers, given other reasonable parameters.
        assert(num_batch_pieces % m_out_split == 0);
        assert(complete_linear_out_size / m_out_split < kMaxBufferSize);
        assert(batch_size % m_out_split == 0);
        m_out_batch_size = batch_size / m_out_split;
        assert(m_out_batch_size % 48 == 0);

        m_model = nn::MetalModel(model_config, chunk_size, batch_size, m_out_split, m_device);
        m_model->load_state_dict(state_dict);
        m_model->eval();

        m_command_queue = m_device->newCommandQueue();
        m_mtl_event = m_device->newSharedEvent();
        m_bwd_scan_cps = make_cps(m_device, "backward_scan", {});
        m_fwd_scan_cps = make_cps(m_device, "forward_scan", {});
        m_add_softmax_cps = make_cps(m_device, "add_softmax", {});

        m_metal_thread.reset(new std::thread(&MetalCaller::metal_thread_fn, this));

        int num_decode_threads = std::max(1, get_apple_cpu_perf_core_count() - 1);
        m_decode_threads.reserve(num_decode_threads);
        for (int i = 0; i < num_decode_threads; ++i) {
            m_decode_threads.emplace_back(new std::thread(&MetalCaller::decode_thread_fn, this, i));
        }

        int T = m_out_chunk_size;
        int C = model_config.outsize;
        int Cs = m_states;

        int y = pow(n_base, model_config.state_len);

        for (int i = 0; i < m_out_split; ++i) {
            m_scores_int8.push_back(torch::empty({T, m_out_batch_size, C}, torch::kInt8));
            m_posts.push_back(torch::empty({m_out_batch_size, T + 1, Cs}));
            m_bwd.push_back(torch::empty({m_out_batch_size, T + 1, Cs}));
        }

        // v3 scores come from a tanh activation whose [-1, 1] range is packed into bytes.
        // The linear kernel scales to [-127, 127] byte range, after which beam search
        // rescales to the expected [-5, 5].
        // v4 scores come from a clamped [-5, 5] range that is rescaled by the kernel to
        // fit into bytes.
        // In both cases beam search applies the same 5/127 factor to scores.
        score_scale = static_cast<float>(5.0 / 127.0);
    }

    ~MetalCaller() {
        std::unique_lock<std::mutex> input_lock(m_input_lock);
        m_terminate = true;
        input_lock.unlock();
        m_input_cv.notify_one();
        m_decode_cv.notify_all();

        m_metal_thread->join();
        for (auto &thr : m_decode_threads) {
            thr->join();
        }
    }

    struct NNTask {
        NNTask(torch::Tensor *input_, int num_chunks_, std::vector<DecodedChunk> *out_chunks_)
                : input(input_), out_chunks(out_chunks_), num_chunks(num_chunks_) {
            static int run = 0;
            run_id = run++;
        }

        torch::Tensor *input;
        std::mutex mut;
        std::condition_variable cv;
        bool ready{false};
        std::vector<DecodedChunk> *out_chunks;
        int num_chunks;
        int decode_chunks_started{0};
        int decode_chunks_finished{0};
        int run_id;
    };

    void call_chunks(torch::Tensor &input, int num_chunks, std::vector<DecodedChunk> &out_chunks) {
        if (num_chunks == 0) {
            return;
        }

        NNTask task(&input, num_chunks, &out_chunks);
        {
            std::lock_guard<std::mutex> lock(m_input_lock);
            m_input_queue.push_front(&task);
        }
        m_input_cv.notify_one();

        std::unique_lock lock(task.mut);
        while (task.decode_chunks_finished != num_chunks) {
            task.cv.wait(lock);
        }
    }

    void metal_thread_fn() {
        while (true) {
            std::unique_lock<std::mutex> input_lock(m_input_lock);
            while (m_input_queue.empty() && !m_terminate) {
                m_input_cv.wait_for(input_lock, 100ms);
            }
            // TODO: finish work before terminating?
            if (m_terminate) {
                return;
            }

            NNTask *task = m_input_queue.back();
            m_input_queue.pop_back();
            input_lock.unlock();

            // TODO: find a more robust way of dealing with Metal kernel launch issues
            for (int try_count = 0; try_count < 5; ++try_count) {
                // The linear layer should not execute until the previous batch has been decoded,
                // since the same buffers are used for successive batches' scores, fwd/bwd scans.
                MTL::SharedEvent *const linear_hold_off =
                        (task->run_id != 0) ? m_mtl_event : nullptr;
                MTL::CommandBuffer *cb = m_model->forward_async(*task->input, linear_hold_off,
                                                                task->run_id - 1, m_scores_int8);

                // The same buffer is used for the forward scan results and the output of
                // m_add_softmax_cps.
                auto &fwd = m_posts;
                // This stage is operating on the split outputs of the linear layer, so
                // the effective batch size is m_out_batch_size.
                std::vector<int32_t> scan_args_{m_out_chunk_size, m_out_batch_size, m_states};
                auto scan_args = create_vec_buffer(m_device, scan_args_);

                for (int i = 0; i < m_out_split; ++i) {
                    // TODO: optimise grid size
                    launch_kernel_no_wait(m_fwd_scan_cps, cb,
                                          {scan_args, mtl_for_tensor(m_scores_int8.at(i)),
                                           mtl_for_tensor(fwd.at(i))},
                                          {}, m_out_batch_size, m_states);

                    launch_kernel_no_wait(m_bwd_scan_cps, cb,
                                          {scan_args, mtl_for_tensor(m_scores_int8.at(i)),
                                           mtl_for_tensor(m_bwd.at(i))},
                                          {}, m_out_batch_size, m_states);

                    launch_kernel_no_wait(
                            m_add_softmax_cps, cb,
                            {scan_args, mtl_for_tensor(fwd.at(i)), mtl_for_tensor(m_bwd.at(i))}, {},
                            m_out_batch_size, m_states);
                }
                if (finishCommandBuffer("linear/scan/softmax", cb, try_count)) {
                    break;
                }
                std::this_thread::sleep_for(20ms);
            }

            // Pass task on to decode threads
            std::unique_lock<std::mutex> decode_lock(m_decode_lock);
            m_decode_queue.push_front(task);
            decode_lock.unlock();
            m_decode_cv.notify_all();
        }
    }

    void decode_thread_fn(int thread_id) {
        while (true) {
            std::unique_lock<std::mutex> decode_lock(m_decode_lock);
            while (m_decode_queue.empty() && !m_terminate) {
                m_decode_cv.wait_for(decode_lock, 100ms);
            }
            // TODO: finish work before terminating?
            if (m_terminate) {
                return;
            }
            NNTask *task = m_decode_queue.back();
            int chunk_idx = task->decode_chunks_started++;
            // If all chunks have been picked up for decoding, remove task from queue
            if (chunk_idx == task->num_chunks - 1) {
                m_decode_queue.pop_back();
            }
            decode_lock.unlock();

            // Model outputs are split across m_out_split buffers.
            assert(m_scores_int8.size() == m_out_split);
            assert(m_bwd.size() == m_out_split);
            assert(m_posts.size() == m_out_split);
            const int out_buf_idx = chunk_idx / m_out_batch_size;
            const int buf_chunk_idx = chunk_idx % m_out_batch_size;

            auto [sequence, qstring, moves] = beam_search_decode(
                    m_scores_int8.at(out_buf_idx).index({Slice(), buf_chunk_idx}),
                    m_bwd.at(out_buf_idx)[buf_chunk_idx], m_posts.at(out_buf_idx)[buf_chunk_idx],
                    m_decoder_options.beam_width, m_decoder_options.beam_cut,
                    m_decoder_options.blank_score, m_decoder_options.q_shift,
                    m_decoder_options.q_scale, m_decoder_options.temperature, score_scale);

            (*task->out_chunks)[chunk_idx] = DecodedChunk{sequence, qstring, moves};

            // Wake the waiting thread which called `call_chunks()` if we're done decoding
            std::unique_lock<std::mutex> task_lock(task->mut);
            bool done = ++(task->decode_chunks_finished) == task->num_chunks;
            task_lock.unlock();
            if (done) {
                m_mtl_event->setSignaledValue(task->run_id);
                task->cv.notify_one();
            }
        }
    }

    bool m_terminate{false};
    std::deque<NNTask *> m_input_queue;
    std::deque<NNTask *> m_decode_queue;
    std::mutex m_input_lock;
    std::condition_variable m_input_cv;
    std::unique_ptr<std::thread> m_metal_thread;
    std::mutex m_decode_lock;
    std::condition_variable m_decode_cv;
    std::vector<std::unique_ptr<std::thread>> m_decode_threads;
    DecoderOptions m_decoder_options;
    nn::MetalModel m_model{nullptr};
    MTL::Device *m_device;
    MTL::CommandQueue *m_command_queue;
    MTL::ComputePipelineState *m_bwd_scan_cps, *m_fwd_scan_cps, *m_add_softmax_cps;
    MTL::SharedEvent *m_mtl_event;
    std::vector<torch::Tensor> m_scores_int8, m_posts, m_bwd;
    int m_out_chunk_size, m_batch_size, m_states, m_model_stride;
    // Number of pieces the linear output is split into, for reasons of
    // buffer size constraints.
    int m_out_split;
    int m_out_batch_size;
    // v3 and v4 models have different score scaling requirements.
    float score_scale{0.0f};
};

std::shared_ptr<MetalCaller> create_metal_caller(const std::filesystem::path &model_path,
                                                 int chunk_size,
                                                 int batch_size) {
    return std::make_shared<MetalCaller>(model_path, chunk_size, batch_size);
}

MetalModelRunner::MetalModelRunner(std::shared_ptr<MetalCaller> caller,
                                   int chunk_size,
                                   int batch_size)
        : m_caller(caller) {
    // adjust chunk size to be a multiple of the stride
    chunk_size -= chunk_size % model_stride();

    m_input = torch::empty({batch_size, 1, chunk_size},
                           torch::TensorOptions().dtype(torch::kF32).device(torch::kCPU));
}

void MetalModelRunner::accept_chunk(int chunk_idx, at::Tensor slice) {
    m_input.index_put_({chunk_idx, 0}, slice);
}

std::vector<DecodedChunk> MetalModelRunner::call_chunks(int num_chunks) {
    std::vector<DecodedChunk> out_chunks(num_chunks);
    m_caller->call_chunks(m_input, num_chunks, out_chunks);
    return out_chunks;
}

size_t MetalModelRunner::model_stride() const { return m_caller->m_model_stride; }
size_t MetalModelRunner::chunk_size() const { return m_input.size(2); }

}  // namespace dorado
