#include "MetalCRFModel.h"

#include "torch_utils/module_utils.h"
#include "utils/math_utils.h"

#include <spdlog/spdlog.h>

#include <stdexcept>

// Splitting up command buffers can be useful since it allows Xcode to make GPU captures.
#define USE_SPLIT_LSTM_COMMAND_BUFFERS 0

namespace {
constexpr int kLstmGates = 4;
// SIMD tile size dictated by the Metal spec.
constexpr int kTileSize = 8;
// We assume non-AMD GPUs, in which case this is 32.
constexpr int kSIMDGroupWidth = 32;

constexpr auto torch_dtype = torch::kF16;
const size_t dtype_bytes = torch::elementSize(torch_dtype);

CREATE_POINT_OF_INTEREST_ID(MetalCRFModel);

}  // namespace

using namespace dorado::utils;
using torch::indexing::Slice;

namespace dorado::basecall::nn {

MetalLinearImpl::MetalLinearImpl(int insize, int outsize, bool has_bias) {
    auto weight = torch::empty({outsize, insize});
    auto bias = torch::empty({outsize});
    register_parameter("weight", weight, false);
    if (has_bias) {
        register_parameter("bias", bias, false);
    }
}

MetalConv1dImpl::MetalConv1dImpl(int layer,
                                 int in_size_,
                                 int out_size_,
                                 int win_size_,
                                 int stride_,
                                 config::Activation activation,
                                 int chunk_size_,
                                 int batch_size_,
                                 MTL::Device *const device)
        : in_size(in_size_),
          out_size(out_size_),
          win_size(win_size_),
          stride(stride_),
          chunk_size(chunk_size_),
          batch_size(batch_size_) {
    if (layer < 1 || layer > 3) {
        throw std::runtime_error("MetalCRFModel invalid config - only expected layers 1-3");
    }

    // For layers 1 and 2 we only have kernels for particular in/out feature sizes.
    if (layer == 1) {
        // Simplex in_size == 1.  Duplex in_size == 13.
        if (!(in_size == 1 || in_size == 13)) {
            throw std::runtime_error(
                    "MetalCRFModel invalid config - layer 1 in_size must be 1 or 13");
        }

        if (!(out_size == 4 || out_size == 16)) {
            throw std::runtime_error(
                    "MetalCRFModel invalid config - layer 1 out_size must be 4 or 16");
        }
        if (win_size != 5) {
            throw std::runtime_error("MetalCRFModel invalid config - layer 1 win_size must be 5");
        }
    } else if (layer == 2) {
        if (!(in_size == 4 || in_size == 16)) {
            throw std::runtime_error(
                    "MetalCRFModel invalid config - layer 2 in_size must be 4 or 16");
        }
        if (out_size != 16) {
            throw std::runtime_error("MetalCRFModel invalid config - layer 2 out_size must be 16");
        }
        if (win_size != 5) {
            throw std::runtime_error("MetalCRFModel invalid config - layer 2 win_size must be 5");
        }
    }

    if (layer != 3) {
        // Layers 1 and 2 are executed with a single kernel launch.
        // The last 2 arguments are unused.
        const std::vector<int32_t> args{in_size,    win_size,   out_size, stride, win_size / 2,
                                        chunk_size, batch_size, 0,        0};
        auto &buffer = m_args.emplace_back(create_vec_buffer(device, args));
        name_mtl_object(buffer, "conv_args");
    } else {
        // We cut up the time span for individual kernel launches for conv3 since it is by far
        // the most time consuming, and sup times can be of the order of seconds, which
        // is known to provoke command buffer errors.
        // The last 2 arguments specify the output time step range, i.e. time step range after
        // dividing by stride.
        const int output_time_step_count = chunk_size / stride;
        constexpr int kMaxTimeSteps = 20;
        const int num_pieces = utils::div_round_up(output_time_step_count, kMaxTimeSteps);
        for (int i = 0; i < num_pieces; ++i) {
            const int time_step_begin = i * kMaxTimeSteps;
            const int time_step_end = std::min((i + 1) * kMaxTimeSteps, output_time_step_count);
            const std::vector<int32_t> args{in_size,    win_size,        out_size,
                                            stride,     win_size / 2,    chunk_size,
                                            batch_size, time_step_begin, time_step_end};
            auto &buffer = m_args.emplace_back(create_vec_buffer(device, args));
            name_mtl_object(buffer, fmt::format("conv_args_{}", i).c_str());
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
    name_mtl_object(mtl_for_tensor(t_weights_bias),
                    fmt::format("conv{}_weights_bias", layer).c_str());

    kernel_simd_groups = (layer == 3 || (layer == 2 && in_size == 16)) ? 4 : 16;
    kernel_thread_groups = get_mtl_device_core_count() * 4;

    std::vector<std::tuple<std::string, MetalConstant>> metal_constants = {
            {"kConvOutputClamp", activation == config::Activation::SWISH_CLAMP},
            {"kConvTanhActivation", activation == config::Activation::TANH},
    };
    const int kernel_threads = kSIMDGroupWidth * kernel_simd_groups;
    std::string kernel_name = "conv" + std::to_string(layer);
    // Layer 1 and 2 conv kernels are tailored to specific feature sizes.
    if (layer == 1 || layer == 2) {
        kernel_name += "_in" + std::to_string(in_size) + "_out" + std::to_string(out_size);
    }

    conv_cps = make_cps(device, kernel_name + "_simd", metal_constants, kernel_threads);
}

void MetalConv1dImpl::run(MTL::CommandQueue *command_queue,
                          MTL::Buffer *mat_in,
                          MTL::Buffer *mat_out) {
    for (const auto &args : m_args) {
        std::vector<MTL::Buffer *> buffers{args.get(), mat_in, mtl_for_tensor(t_weights_bias),
                                           mat_out};
        launch_kernel(conv_cps.get(), command_queue, buffers, {}, kernel_thread_groups,
                      kernel_simd_groups * kSIMDGroupWidth);
    }
}

void MetalConv1dImpl::run(MTL::CommandBuffer *command_buffer,
                          MTL::Buffer *mat_in,
                          MTL::Buffer *mat_out) {
    for (const auto &args : m_args) {
        std::vector<MTL::Buffer *> buffers{args.get(), mat_in, mtl_for_tensor(t_weights_bias),
                                           mat_out};
        launch_kernel_no_wait(conv_cps.get(), command_buffer, buffers, {}, kernel_thread_groups,
                              kernel_simd_groups * kSIMDGroupWidth);
    }
}

void MetalConv1dImpl::load_weights() {
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

MetalLSTMImpl::MetalLSTMImpl(int layer_size, bool reverse_) : reverse(reverse_) {
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
    name_mtl_object(mtl_for_tensor(t_weights_bias), "lstm_weights_bias");
}

MetalBlockImpl::MetalBlockImpl(int chunk_size_,
                               int batch_size_,
                               const config::BasecallModelConfig &config_,
                               int out_split_,
                               MTL::Device *const device)
        : m_device(device), in_chunk_size(chunk_size_), batch_size(batch_size_), config(config_) {
    m_command_queue = NS::TransferPtr(m_device->newCommandQueue());

    lstm_chunk_size = in_chunk_size / config.stride;

    // args for LSTM kernel
    {
        // We enforce a maximum time step count for each LSTM kernel because long running
        // kernels increase the likelihood of command buffer submission errors, of various
        // types.  Each args buffer here is for a different time step range.
        constexpr int kMaxTimeSteps = 20;
        const int num_pieces = utils::div_round_up(lstm_chunk_size, kMaxTimeSteps);
        for (int i = 0; i < num_pieces; ++i) {
            const int time_step_begin = i * kMaxTimeSteps;
            const int time_step_end = std::min((i + 1) * kMaxTimeSteps, lstm_chunk_size);
            std::vector<int32_t> args{batch_size / kTileSize, lstm_chunk_size, time_step_begin,
                                      time_step_end};
            auto &buffer = m_args_lstm.emplace_back(create_vec_buffer(m_device, args));
            name_mtl_object(buffer, fmt::format("lstm_args_{}", i).c_str());
        }
        spdlog::debug("lstm_chunk_size {} => {} LSTM kernel launches", lstm_chunk_size, num_pieces);
    }

    // args for final (possibly only) linear layer kernel.
    // Each output buffer requires a distinct input offset, so we must have a separate args buffer.
    args_linear.resize(out_split_);
    const int32_t in_batch_tiles = batch_size / kTileSize;
    const int32_t out_batch_tiles = (batch_size / out_split_) / kTileSize;
    for (int i = 0; i < out_split_; ++i) {
        const int32_t in_batch_tile_offset = out_batch_tiles * i;
        std::vector<int32_t> args_linear_ = {in_batch_tiles, in_batch_tile_offset, out_batch_tiles,
                                             lstm_chunk_size};
        auto &buffer = args_linear.at(i);
        buffer = create_vec_buffer(m_device, args_linear_);
        name_mtl_object(buffer, fmt::format("linear_args_{}", i).c_str());
    }
    args_linear2 = create_vec_buffer<int32_t>(
            device, {out_batch_tiles, 0, out_batch_tiles, lstm_chunk_size});
    name_mtl_object(args_linear2, "linear2_args");

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
        kernel_simd_groups = 24;  // Note - we may want to set this to 32 for M4
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
    lstm_cps[0] = make_cps(m_device, "lstm",
                           {
                                   {"kLstmLayerSize", config.lstm_size},
                                   {"kLstmReversedInTime", false},
                           },
                           lstm_threads);
    lstm_cps[1] = make_cps(m_device, "lstm",
                           {
                                   {"kLstmLayerSize", config.lstm_size},
                                   {"kLstmReversedInTime", true},
                           },
                           lstm_threads);

    // The temp buffer used for these purposes (number of elements of `torch_dtype` in []):
    // - Store output of second conv layer [in_chunk_size * batch_size * kMaxConv2OutChannels]
    // - Store output of first linear layer if there are two
    //   [lstm_chunk_size * batch_size * decomposition / out_split_]
    // We size mat_temp here for conv2 output, potentially increasing it below in the case of a
    // linear decomposition model.
    constexpr int kMaxConv2OutChannels = 16;
    int mat_temp_elems = batch_size * kMaxConv2OutChannels * in_chunk_size;

    assert(config.convs.size() == 3);
    const auto cv1 = config.convs[0];
    conv1 = register_module("conv1",
                            MetalConv1d(1, config.num_features, cv1.size, cv1.winlen, cv1.stride,
                                        cv1.activation, in_chunk_size, batch_size, device));
    const auto cv2 = config.convs[1];
    conv2 = register_module(
            "conv2", MetalConv1d(2, cv1.size, cv2.size, cv2.winlen, cv2.stride, cv2.activation,
                                 in_chunk_size, batch_size, device));
    const auto cv3 = config.convs[2];
    conv3 = register_module(
            "conv3", MetalConv1d(3, cv2.size, cv3.size, cv3.winlen, cv3.stride, cv3.activation,
                                 in_chunk_size, batch_size, device));

    if (cv3.size != config.lstm_size) {
        throw std::runtime_error("MetalCRFModel invalid config - conv_3.size != config.lstm_size");
    }
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
        const auto linear_constants1 = std::vector<std::tuple<std::string, MetalConstant>>({
                {"kLinearInSize", config.lstm_size},
                {"kLinearOutSize", decomposition},
                {"kLinearOutputScale", 1.0f},
                {"kLinearOutputClamp", false},
                {"kLinearOutputTanh", false},
                {"kLinearOutputAsByte", false},
        });
        linear_cps[0] =
                make_cps(m_device, "linear_from_rev_lstm", linear_constants1, linear_threads);
        const auto linear_constants2 = std::vector<std::tuple<std::string, MetalConstant>>({
                {"kLinearInSize", decomposition},
                {"kLinearOutSize", config.outsize},
                // Rescale from clamped [-5.0, 5.0] range to byte range.
                {"kLinearOutputScale", 127.0f / 5.0f},
                {"kLinearOutputClamp", true},
                {"kLinearOutputTanh", false},
                {"kLinearOutputAsByte", true},
        });
        linear_cps[1] = make_cps(m_device, "linear", linear_constants2, linear_threads);
        // We use mat_temp for the output of the first linear layer, so ensure it is large
        // enough for that purpose.
        mat_temp_elems = std::max(mat_temp_elems,
                                  decomposition * (batch_size / out_split_) * lstm_chunk_size);
    } else {
        const bool is_v3_model = (config.num_features == 1 && cv1.size == 4) ||
                                 (config.num_features == 13 && cv1.size == 16);
        const auto linear_constants = std::vector<std::tuple<std::string, MetalConstant>>({
                {"kLinearInSize", config.lstm_size},
                {"kLinearOutSize", config.outsize},
                // If v4, rescale from clamped [-5.0, 5.0] range to byte range.
                // If v3, rescale from tanh [-1.0, 1,0] range to byte range.
                {"kLinearOutputScale", is_v3_model ? 127.0f : (127.0f / 5.0f)},
                {"kLinearOutputClamp", !is_v3_model},
                {"kLinearOutputTanh", is_v3_model},
                {"kLinearOutputAsByte", true},
        });
        linear_cps[0] =
                make_cps(m_device, "linear_from_rev_lstm", linear_constants, linear_threads);
        // Single matmul that may or may not have a bias.
        if (!config.out_features.has_value()) {
            linear1 = register_module("linear1",
                                      MetalLinear(config.lstm_size, config.outsize, config.bias));
        }
    }

    // This buffer is used for several layers of the model.
    mat_working_mem = create_buffer(
            m_device, size_t(lstm_chunk_size + 3) * batch_size * config.lstm_size * dtype_bytes);
    mat_state = create_buffer(m_device, batch_size * config.lstm_size * dtype_bytes);
    mat_temp = create_buffer(m_device, mat_temp_elems * dtype_bytes);
    name_mtl_object(mat_working_mem, "mat_working_mem");
    name_mtl_object(mat_state, "mat_state");
    name_mtl_object(mat_temp, "mat_temp");
}

void MetalBlockImpl::load_weights() {
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
        name_mtl_object(mtl_for_tensor(rnn->t_weights_bias), "rnn_weights");
    }

    // Load and prepare linear layer weights.
    auto get_linear_weights = [](MetalLinear &linear, bool use_bias) -> NS::SharedPtr<MTL::Buffer> {
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
MTL::CommandBuffer *MetalBlockImpl::forward_async(at::Tensor &in,
                                                  MTL::SharedEvent *const linear_hold_off_event,
                                                  uint64_t linear_hold_off_id,
                                                  int try_count,
                                                  std::vector<at::Tensor> &out) {
    {
        POINT_OF_INTEREST_SCOPE(MetalCRFModel, convolutions, "try_count=%i", try_count);
        auto command_buffer = next_command_buffer(m_command_queue.get(), try_count);

        if (in.dtype() != torch::kF16) {
            throw std::runtime_error("Input tensor must be float16.");
        }
        conv1->run(command_buffer, mtl_for_tensor(in), mat_working_mem.get());
        conv2->run(command_buffer, mat_working_mem.get(), mat_temp.get());
        conv3->run(command_buffer, mat_temp.get(), mat_working_mem.get());
        if (!run_command_buffer("convolutions", command_buffer, try_count)) {
            return nullptr;
        }
    }

    std::string lstm_label = "lstm_rnn0";
    for (auto &rnn : {rnn1, rnn2, rnn3, rnn4, rnn5}) {
        lstm_label.back()++;
        POINT_OF_INTEREST_SCOPE(MetalCRFModel, lstm_layer, "id=%s, try_count=%i",
                                lstm_label.c_str(), try_count);

#if !USE_SPLIT_LSTM_COMMAND_BUFFERS
        auto *command_buffer = next_command_buffer(m_command_queue.get(), try_count);
#endif

        const int kResBufSize =
                static_cast<int>(dtype_bytes * kernel_simd_groups * 2 * kTileSize * kTileSize);
        const int kOutBufSize =
                static_cast<int>(dtype_bytes * kernel_simd_groups * kTileSize * kTileSize);
        const std::vector<int> tg_buffer_lens{kResBufSize, kOutBufSize};
        for (const auto &args_lstm : m_args_lstm) {
            const std::vector<MTL::Buffer *> buffers{args_lstm.get(), mat_working_mem.get(),
                                                     mtl_for_tensor(rnn->t_weights_bias),
                                                     mat_state.get()};
#if USE_SPLIT_LSTM_COMMAND_BUFFERS
            auto *command_buffer = next_command_buffer(m_command_queue.get(), try_count);
#endif
            launch_kernel_no_wait(lstm_cps[rnn->reverse].get(), command_buffer, buffers,
                                  tg_buffer_lens, kernel_thread_groups,
                                  kernel_simd_groups * kSIMDGroupWidth);
#if USE_SPLIT_LSTM_COMMAND_BUFFERS
            if (!run_command_buffer(lstm_label.c_str(), command_buffer, try_count)) {
                return nullptr;
            }
#endif
        }

#if !USE_SPLIT_LSTM_COMMAND_BUFFERS
        if (!run_command_buffer(lstm_label.c_str(), command_buffer, try_count)) {
            return nullptr;
        }
#endif
    }

    auto *command_buffer = next_command_buffer(m_command_queue.get(), try_count);

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
            launch_kernel_no_wait(
                    linear_cps[0].get(), command_buffer,
                    {args_buffer, mat_working_mem.get(), linear_weights[0].get(), mat_temp.get()},
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

MetalCRFModelImpl::MetalCRFModelImpl(const config::BasecallModelConfig &config,
                                     int chunk_size,
                                     int batch_size,
                                     int out_split,
                                     MTL::Device *const device) {
    mtl_block = register_module("mtl_block",
                                MetalBlock(chunk_size, batch_size, config, out_split, device));
}

void MetalCRFModelImpl::load_state_dict(const std::vector<at::Tensor> &weights) {
    utils::load_state_dict(*this, weights);
    mtl_block->load_weights();
}

MTL::CommandBuffer *MetalCRFModelImpl::forward_async(at::Tensor &in,
                                                     MTL::SharedEvent *const linear_hold_off_event,
                                                     uint64_t linear_hold_off_id,
                                                     int try_count,
                                                     std::vector<at::Tensor> &out) {
    POINT_OF_INTEREST_SCOPE(MetalCRFModel, forward_async, "try_count=%i", try_count);
    return mtl_block->forward_async(in, linear_hold_off_event, linear_hold_off_id, try_count, out);
}

}  // namespace dorado::basecall::nn
