#include "CRFModel.h"

#include "../utils/models.h"
#include "../utils/module_utils.h"
#include "../utils/tensor_utils.h"

#if DORADO_GPU_BUILD && !defined(__APPLE__)
#include "../utils/cuda_utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

extern "C" {
#include "koi.h"
}

#define USE_CUDA_LSTM 1
#define CUDA_PROFILE_TO_CERR 0
#else
#define USE_CUDA_LSTM 0
#endif

#include <math.h>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>
#include <toml.hpp>
#include <torch/torch.h>

#include <limits>
#include <string>

// Different configurations for running Quantised LSTM
constexpr bool g_options_no_i8 = false;
constexpr bool g_options_no_conv_i8 = true;
constexpr float g_options_conv3_max = 3.5f;

using namespace torch::nn;
namespace F = torch::nn::functional;
using Slice = torch::indexing::Slice;

#if USE_CUDA_LSTM

enum LstmMode { CUBLAS_TN2C, QUANTISED_NTC, CUTLASS_TNC_I8, CUTLASS_TNC_F16 };

static LstmMode get_cuda_lstm_mode(int layer_idx, int layer_size) {
    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
    if (!(prop->major == 6 && prop->minor == 2) &&
        ((layer_size == 96) || (layer_size == 128 && prop->major < 8))) {
        return QUANTISED_NTC;
    } else if (layer_size <= 1024 && (layer_size % 128) == 0 &&
               (prop->major == 8 || prop->major == 9) && prop->minor == 0) {
        if (g_options_no_i8 || (layer_idx == 0 && g_options_no_conv_i8)) {
            return CUTLASS_TNC_F16;
        }
        return CUTLASS_TNC_I8;
    }
    return CUBLAS_TN2C;
}

#endif  // if USE_CUDA_LSTM

#if CUDA_PROFILE_TO_CERR
#define CUDA_CHECK(X)                                                                         \
    {                                                                                         \
        cudaError_t error = X;                                                                \
        if (error != cudaSuccess) {                                                           \
            printf("CUDA returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), \
                   error, __LINE__);                                                          \
            exit(EXIT_FAILURE);                                                               \
        }                                                                                     \
    }

// Times range and prints it to console so you don't have to generate a QDREP file to perform
// basic profiling
class ScopedProfileRange {
public:
    explicit ScopedProfileRange(const char *label) : m_nvtx_range(label), m_label(label) {
        m_stream = at::cuda::getCurrentCUDAStream().stream();
        CUDA_CHECK(cudaEventCreate(&m_start));
        CUDA_CHECK(cudaEventRecord(m_start, m_stream));
        m_active = true;
    }

    ~ScopedProfileRange() { finish(); }

private:
    void finish() {
        if (!m_active) {
            return;
        }
        cudaEvent_t stop;
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(stop, m_stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float timeMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&timeMs, m_start, stop));
        CUDA_CHECK(cudaEventDestroy(m_start));
        CUDA_CHECK(cudaEventDestroy(stop));
        std::cerr << "[" << m_label << " " << timeMs << " ms]" << std::endl;
        m_active = false;
    }

    nvtx3::scoped_range m_nvtx_range;
    const char *m_label;
    cudaStream_t m_stream;
    cudaEvent_t m_start;
    bool m_active;
};
#else  // if CUDA_PROFILE_TO_CERR
using ScopedProfileRange = nvtx3::scoped_range;
#endif

namespace {
template <class Model>
ModuleHolder<AnyModule> populate_model(Model &&model,
                                       const std::filesystem::path &path,
                                       const torch::TensorOptions &options,
                                       bool decomposition,
                                       bool linear_layer_bias) {
    auto state_dict = dorado::load_crf_model_weights(path, decomposition, linear_layer_bias);
    model->load_state_dict(state_dict);
    model->to(options.dtype_opt().value().toScalarType());
    model->to(options.device_opt().value());
    model->eval();

    auto module = AnyModule(model);
    auto holder = ModuleHolder<AnyModule>(module);
    return holder;
}
}  // namespace

namespace dorado {

namespace nn {

static constexpr float SWISH_LOWER_BOUND = -0.278464543f;  // global minimum of `x * sigmoid(x)`
static constexpr float I8_RANGE = 127.f;

struct ConvolutionImpl : Module {
    ConvolutionImpl(int size,
                    int outsize,
                    int k,
                    int stride_,
                    bool clamp_,
                    float max_value_,
                    bool to_lstm_)
            : in_size(size),
              out_size(outsize),
              window_size(k),
              stride(stride_),
              clamp(clamp_),
              max_value(clamp_ ? max_value_ : std::numeric_limits<float>::max()),
              to_lstm(to_lstm_) {
        conv = register_module(
                "conv", Conv1d(Conv1dOptions(size, outsize, k).stride(stride).padding(k / 2)));
        activation = register_module("activation", SiLU());
    }

    torch::Tensor forward(torch::Tensor x) {
        // Input x is [N, C_in, T_in], contiguity optional
#if USE_CUDA_LSTM
        if (to_lstm && x.device() != torch::kCPU) {
            c10::cuda::CUDAGuard device_guard(x.device());
            auto stream = at::cuda::getCurrentCUDAStream().stream();

            int batch_size = x.size(0);
            int chunk_size_in = x.size(2);
            int chunk_size_out = chunk_size_in / stride;
            auto w_device = conv->weight.view({out_size, in_size * window_size})
                                    .t()
                                    .to(x.options())
                                    .contiguous();
            auto b_device = conv->bias.to(x.options());

            auto lstm_mode = get_cuda_lstm_mode(0, out_size);
            if (lstm_mode == QUANTISED_NTC) {
                auto res = torch::empty({batch_size, chunk_size_out, out_size}, x.options());
                auto res_2D = res.view({-1, out_size});
                auto ntcw_mat = torch::empty({batch_size, chunk_size_out, in_size, window_size},
                                             x.options());
                host_window_ntcw_f16(stream, x.stride(0), x.stride(2), x.stride(1), batch_size,
                                     chunk_size_in, in_size, window_size, stride,
                                     ntcw_mat.stride(0), ntcw_mat.stride(1), ntcw_mat.stride(2),
                                     ntcw_mat.stride(3), x.data_ptr(), ntcw_mat.data_ptr());
                dorado::utils::matmul_f16(ntcw_mat.view({-1, in_size * window_size}), w_device,
                                          res_2D);
                host_bias_swish_f16_clamp(stream, res_2D.size(0), res_2D.size(1), res_2D.stride(0),
                                          res_2D.data_ptr(), b_device.data_ptr(), max_value);
                // Output is [N, T_out, C_out], F16, contiguous
                return res;
            } else {
                torch::Tensor res, mm_out;
                if (lstm_mode == CUTLASS_TNC_I8) {
                    // Output is [T_out + 3, N, C_out], I8, non-contiguous (`stride(2)` is
                    // doubled)
                    auto res_f16 =
                            torch::empty({chunk_size_out + 3, batch_size, out_size}, x.options());
                    res = res_f16.view(torch::kI8).slice(2, 0, out_size);
                    mm_out = res_f16.slice(0, 1, chunk_size_out + 1);
                } else if (lstm_mode == CUTLASS_TNC_F16) {
                    // Output is [T_out + 3, N, C_out], F16, contiguous
                    res = torch::empty({chunk_size_out + 3, batch_size, out_size}, x.options());
                    mm_out = res.slice(0, 1, chunk_size_out + 1).view({-1, out_size});
                } else if (lstm_mode == CUBLAS_TN2C) {
                    // Output is [T_out + 1, N, 2, C_out], F16, contiguous
                    res = torch::empty({chunk_size_out + 1, batch_size, 2, out_size}, x.options());
                    auto res_TNC = res.slice(0, 1, chunk_size_out + 1).select(2, 1);
                    mm_out = res_TNC.view({-1, out_size});
                }

                auto tncw_mat = torch::empty({chunk_size_out, batch_size, in_size, window_size},
                                             x.options());
                host_window_ntcw_f16(stream, x.stride(0), x.stride(2), x.stride(1), batch_size,
                                     chunk_size_in, in_size, window_size, stride,
                                     tncw_mat.stride(1), tncw_mat.stride(0), tncw_mat.stride(2),
                                     tncw_mat.stride(3), x.data_ptr(), tncw_mat.data_ptr());
                dorado::utils::matmul_f16(tncw_mat.view({-1, in_size * window_size}), w_device,
                                          mm_out);
                host_bias_swish_f16_clamp(stream, mm_out.size(0), mm_out.size(1), mm_out.stride(0),
                                          mm_out.data_ptr(), b_device.data_ptr(), max_value);

                if (lstm_mode == CUTLASS_TNC_I8) {
                    // Convert F16 to I8 in-place. Each C_out-wide F16 vector gets converted such
                    // that its first half contains the I8 data while its second half is unused.
                    host_convert_inplace(stream, mm_out.data_ptr(), KOI_F16, KOI_I8,
                                         mm_out.stride(0), 0, mm_out.size(0), mm_out.size(1));
                }

                // Zero-fill the timesteps representing initial LSTM input (in both directions)
                if (lstm_mode == CUBLAS_TN2C) {
                    res[0].select(1, 1) = 0;
                    res[-1].select(1, 0) = 0;
                } else {
                    res[0] = 0;
                    res[-1] = 0;
                }
                return res;
            }
        }
#endif
        // Alternative route - non CUDALSTM route.
        x = activation(conv(x));
        if (clamp) {
            x.clamp_(c10::nullopt, max_value);
        }
        if (to_lstm) {
            // Output is [N, T_out, C_out], non-contiguous
            return x.transpose(1, 2);
        } else {
            // Output is [N, C_out, T_out], contiguous
            return x;
        }
    }

    Conv1d conv{nullptr};
    SiLU activation{nullptr};
    int in_size;
    int out_size;
    int window_size;
    int stride;
    const bool clamp;
    const float max_value;
    const bool to_lstm;
};

struct LinearCRFImpl : Module {
    LinearCRFImpl(int insize, int outsize) : scale(5), blank_score(2.0), expand_blanks(false) {
        linear = register_module("linear", Linear(insize, outsize));
        activation = register_module("activation", Tanh());
    };

    torch::Tensor forward(torch::Tensor x) {
        // Input x is [N, T, C], contiguity optional
        auto N = x.size(0);
        auto T = x.size(1);

        torch::Tensor scores;
#if USE_CUDA_LSTM
        if (x.device() != torch::kCPU) {
            // Optimised version of the else branch for CUDA devices
            c10::cuda::CUDAGuard device_guard(x.device());
            auto stream = at::cuda::getCurrentCUDAStream().stream();

            x = x.contiguous().reshape({N * T, -1});
            scores = torch::matmul(x, linear->weight.t());
            host_bias_tanh_scale_f16(stream, N * T, scores.size(1), scale, scores.data_ptr(),
                                     linear->bias.data_ptr());
            scores = scores.view({N, T, -1});
        } else
#endif  // if USE_CUDA_LSTM
        {
            scores = activation(linear(x)) * scale;
        }

        if (expand_blanks == true) {
            scores = scores.contiguous();
            int C = scores.size(2);
            scores = F::pad(scores.view({N, T, C / 4, 4}),
                            F::PadFuncOptions({1, 0, 0, 0, 0, 0, 0, 0}).value(blank_score))
                             .view({N, T, -1});
        }

        // Output is [N, T, C], contiguous
        return scores;
    }

    int scale;
    int blank_score;
    bool expand_blanks;
    Linear linear{nullptr};
    Tanh activation{nullptr};
};

struct LSTMStackImpl : Module {
    LSTMStackImpl(int size) : layer_size(size) {
        // torch::nn::LSTM expects/produces [N, T, C] with batch_first == true
        rnn1 = register_module("rnn1", LSTM(LSTMOptions(size, size).batch_first(true)));
        rnn2 = register_module("rnn2", LSTM(LSTMOptions(size, size).batch_first(true)));
        rnn3 = register_module("rnn3", LSTM(LSTMOptions(size, size).batch_first(true)));
        rnn4 = register_module("rnn4", LSTM(LSTMOptions(size, size).batch_first(true)));
        rnn5 = register_module("rnn5", LSTM(LSTMOptions(size, size).batch_first(true)));
    };

    // Dispatch to different forward method depending on whether we use quantized LSTMs or not
    torch::Tensor forward(torch::Tensor x) {
        // Input x is [N, T, C], contiguity optional
#if USE_CUDA_LSTM
        if (x.is_cuda()) {
            c10::cuda::CUDAGuard device_guard(x.device());
            ScopedProfileRange spr("lstm_stack");

            auto mode = get_cuda_lstm_mode(0, layer_size);
            if (mode == QUANTISED_NTC) {
                // Output is [N, T, C], contiguous
                return forward_quantized(x);
            } else if (mode == CUBLAS_TN2C) {
                // Output is [N, T, C], non-contiguous
                return forward_cublas(x);
            } else {
                // Output is [N, T, C], non-contiguous
                return forward_cutlass(x);
            }
        }
#endif
        auto [y1, h1] = rnn1(x.flip(1));
        auto [y2, h2] = rnn2(y1.flip(1));
        auto [y3, h3] = rnn3(y2.flip(1));
        auto [y4, h4] = rnn4(y3.flip(1));
        auto [y5, h5] = rnn5(y4.flip(1));

        // Output is [N, T, C], non-contiguous
        return y5.flip(1);
    }

#if USE_CUDA_LSTM
private:
    torch::Tensor forward_cublas(torch::Tensor in) {
        // input is ([T+1, N, 2, C], contiguous) (see below)
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        const int chunk_size = in.size(0) - 1;
        const int batch_size = in.size(1);
        assert(layer_size == in.size(3));
        assert(in.dim() == 4 && in.size(2) == 2);
        assert(in.dtype() == torch::kF16);
        assert(in.is_contiguous());

        // Working memory is laid out as [T+1][N][2][C] in memory, where the 2 serves to
        // interleave input and output for each LSTM layer in a specific way. The reverse LSTM
        // layers (rnn1, rnn3, rnn5) use right as input and left as output, whereas the forward
        // LSTM layers (rnn2, rnn4) use left as input and right as output.
        //
        // The interleaving means that x(t) and h(t-1), i.e. the input for the current timestep
        // and the output of the previous timestep, appear concatenated in memory and we can
        // perform a single matmul with the concatenated WU matrix
        // Note that both working_mem[chunk_size][:][0][:] and working_mem[0][:][1][:] remain
        // all zeroes, representing the initial LSTM state h(-1) in either direction.
        auto inout_all = in.view({chunk_size + 1, batch_size, -1});
        auto inout_left = in.slice(0, 0, chunk_size).select(2, 0);
        auto inout_right = in.slice(0, 1, chunk_size + 1).select(2, 1);
        auto gate_buf = torch::empty({batch_size, layer_size * 4}, in.options());

        int layer_idx = 0;
        for (auto &rnn : {rnn1, rnn2, rnn3, rnn4, rnn5}) {
            bool reverse = !(layer_idx & 1);
            ScopedProfileRange spr_lstm("lstm_layer");
            auto state_buf = torch::zeros({batch_size, layer_size}, in.options());
            {
                if (int(device_weights.size()) == layer_idx) {
                    auto const &params = rnn->named_parameters();
                    auto w_ih = params["weight_ih_l0"];
                    auto w_hh = params["weight_hh_l0"];
                    device_bias.push_back(params["bias_ih_l0"].to(in.options()));
                    auto weights = torch::cat({reverse ? w_hh : w_ih, reverse ? w_ih : w_hh}, 1);
                    device_weights.push_back(weights.t().contiguous().to(in.options()));
                }
                for (int ts = 0; ts < chunk_size; ++ts) {
                    auto timestep_in = inout_all[reverse ? (chunk_size - ts) : ts];
                    auto timestep_out = reverse ? inout_left[chunk_size - ts - 1] : inout_right[ts];
                    // Timestep matrix mulitplication
                    dorado::utils::matmul_f16(timestep_in, device_weights[layer_idx], gate_buf);
                    host_lstm_step_f16(stream, batch_size, layer_size,
                                       device_bias[layer_idx].data_ptr(), gate_buf.data_ptr(),
                                       state_buf.data_ptr(), timestep_out.data_ptr());
                }
            }
            ++layer_idx;
        }

        // Output is [N, T, C], non-contiguous
        return inout_left.transpose(1, 0);
    }

    torch::Tensor forward_cutlass(torch::Tensor in) {
#ifdef DORADO_TX2  // Koi for TX2 does not have Cutlass kernels
        throw std::logic_error("No Cutlass kernels in Jetson TX2 build.");
        return in;
#else
        // input is ([T+3, N, C], contiguous) (see below)
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        const int chunk_size = in.size(0) - 3;
        const int batch_size = in.size(1);
        assert(layer_size == in.size(2));
        assert(in.dim() == 3);
        assert(in.dtype() == torch::kF16 || (use_int8 && in.dtype() == torch::kInt8));
        auto opts_f16 = in.options().dtype(torch::kF16);
        auto opts_i32 = in.options().dtype(torch::kI32);

        // Working memory is laid out as [T+3][N][C] in memory, where the reverse LSTM
        // layers (rnn1, rnn3, rnn5) use [1:-2] as input and [2:-1] as output, whereas the
        // forward LSTM layers (rnn2, rnn4) use [2:-1] as input and [1:-2] as output.
        // Note that both inout[0] and inout[-1] remain all zeroes, representing the initial
        // LSTM state h(-1) in either direction.

        // F16 and Int8 tensors share the same memory. We can convert in-place,
        // doubling stride(2) for the Int8 tensor.
        torch::Tensor inout_all_f16, inout_all_i8;
        int convert_to_int8_layer_idx = -1;
        if (in.dtype() == torch::kF16) {
            inout_all_i8 = in.view(torch::kI8).slice(2, 0, layer_size);
            inout_all_f16 = in;
            convert_to_int8_layer_idx =
                    g_options_no_i8 ? 6 : 0;  // convert after first layer, or never
        } else if (in.dtype() == torch::kInt8) {
            inout_all_i8 = in;
            inout_all_f16 = in.view(torch::kF16);
        }

        // These illustrate the memory layout of the inputs to the fwd and reverse layers (not used)
        //    auto in_rev_i8 = inout_all_i8.slice(0, 1, chunk_size + 1);
        //    auto in_fwd_i8 = inout_all_i8.slice(0, 2, chunk_size + 2);
        //    auto in_rev_f16 = inout_all_f16.slice(0, 1, chunk_size + 1);
        auto in_fwd_f16 = inout_all_f16.slice(0, 2, chunk_size + 2);

        int layer_idx = 0;
        for (auto &rnn : {rnn1, rnn2, rnn3, rnn4, rnn5}) {
            bool reverse = !(layer_idx & 1);
            ScopedProfileRange spr_lstm("lstm_layer");
            auto state_buf = torch::zeros({batch_size, layer_size}, opts_f16);
            auto workspace_buf = torch::empty({1024}, opts_i32);
            auto type_id = (layer_idx > convert_to_int8_layer_idx) ? KOI_I8 : KOI_F16;
            int interleave = 0;  //(type_id == KOI_I8) ? 64 : 32;

            if (int(device_weights.size()) == layer_idx) {
                auto const &params = rnn->named_parameters();
                auto w_ih = params["weight_ih_l0"].to(opts_f16);
                auto w_hh = params["weight_hh_l0"].to(opts_f16);
                auto weights_cpu = torch::cat({reverse ? w_hh : w_ih, reverse ? w_ih : w_hh}, 1);
                auto bias_cpu = params["bias_ih_l0"].to(opts_f16);

                auto layer_device_bias =
                        params["bias_ih_l0"].to(opts_f16).view({4, layer_size}).t();
                if (type_id == KOI_I8) {
                    auto weights_f32 = weights_cpu.t().to(torch::kF32);
                    auto [scale, quantized] = quantize_tensor(weights_f32);
                    weights_cpu = quantized.t();
                    scale = scale.view({4, layer_size}).t();
                    device_scale.push_back(scale.to(opts_f16).contiguous());
                } else {
                    device_scale.push_back(torch::ones_like(layer_device_bias));
                }
                device_bias.push_back(layer_device_bias.contiguous());
                // Cutlass kernel expects weights reordered as <igigigigfofofofo>
                auto weights_cpu_cutlass = torch::empty_like(weights_cpu);
                for (int i = 0; i < layer_size; ++i) {
                    int i0 = i / 4;
                    int i1 = i % 4;
                    weights_cpu_cutlass[i0 * 16 + i1 * 2 + 0] = weights_cpu[i + 0 * layer_size];
                    weights_cpu_cutlass[i0 * 16 + i1 * 2 + 1] = weights_cpu[i + 1 * layer_size];
                    weights_cpu_cutlass[i0 * 16 + i1 * 2 + 8] = weights_cpu[i + 2 * layer_size];
                    weights_cpu_cutlass[i0 * 16 + i1 * 2 + 9] = weights_cpu[i + 3 * layer_size];
                }
                if (interleave) {
                    weights_cpu_cutlass = weights_cpu_cutlass.view({4 * layer_size, -1, interleave})
                                                  .permute({1, 0, 2});
                }
                device_weights.push_back(weights_cpu_cutlass.contiguous().to(in.device()));
            }

            auto in = (type_id == KOI_I8) ? inout_all_i8 : inout_all_f16;
            host_cutlass_lstm(stream, type_id, layer_idx, batch_size, layer_size, chunk_size,
                              reverse ? -1 : 1, in.stride(1), in.data_ptr(),
                              device_weights[layer_idx].data_ptr(),
                              device_bias[layer_idx].data_ptr(), device_scale[layer_idx].data_ptr(),
                              state_buf.data_ptr(), workspace_buf.data_ptr(), interleave, 0);

            if (layer_idx == convert_to_int8_layer_idx) {
                ScopedProfileRange spr_convert("f16_to_int8");
                host_convert_inplace(stream, in.data_ptr(), KOI_F16, KOI_I8, in.stride(1), 0,
                                     in.size(0) * inout_all_f16.size(1), in.size(2));
            }

            ++layer_idx;
        }

        if (!g_options_no_i8) {
            ScopedProfileRange spr_convert("int8_to_f16");
            // in_fwd_f16 shares storage with in_fwd_i8
            host_convert_inplace(stream, in_fwd_f16.data_ptr(), KOI_I8, KOI_F16,
                                 in_fwd_f16.stride(1), 0, in_fwd_f16.size(0) * in_fwd_f16.size(1),
                                 in_fwd_f16.size(2));
        }
        // Output is [N, T, C], non-contiguous
        return in_fwd_f16.transpose(1, 0);
#endif  // ifdef DORADO_TX2 else
    }

    void rearrange_individual_weights(torch::Tensor buffer) {
        torch::Tensor tmp = torch::empty_like(buffer);
        int layer_width = tmp.size(0) / 4;

        //Mapping of LSTM gate weights from IFGO to GIFO order.
        std::vector<std::pair<int, int>> idxs = {std::make_pair(0, 2), std::make_pair(1, 0),
                                                 std::make_pair(2, 1), std::make_pair(3, 3)};

        for (auto idx : idxs) {
            int start_idx = idx.second * layer_width;
            int end_idx = start_idx + layer_width;
            tmp.index({torch::indexing::Slice(idx.first * layer_width,
                                              (idx.first + 1) * layer_width)}) =
                    buffer.index({torch::indexing::Slice(start_idx, end_idx)});
        }

        buffer.index({torch::indexing::Slice()}) = tmp;
    }

    std::pair<torch::Tensor, torch::Tensor> quantize_tensor(torch::Tensor tensor,
                                                            int levels = 256) {
        //Quantize a tensor to int8, returning per-channel scales and the quantized tensor
        //if weights have not been quantized we get some scaling
        auto fp_max = torch::abs(std::get<0>(torch::max(tensor, 0)));
        auto fp_min = torch::abs(std::get<0>(torch::min(tensor, 0)));

        auto fp_range =
                std::get<0>(
                        torch::cat(
                                {fp_min.index({torch::indexing::Slice(), torch::indexing::None}),
                                 fp_max.index({torch::indexing::Slice(), torch::indexing::None})},
                                1)
                                .max(1)) *
                2;
        auto quantization_scale = levels / fp_range;
        auto quantization_max = (levels / 2) - 1;

        auto tensor_quantized = (tensor * quantization_scale)
                                        .round()
                                        .clip(-quantization_max, quantization_max)
                                        .to(torch::kI8);

        return std::pair<torch::Tensor, torch::Tensor>(quantization_scale.to(torch::kFloat32),
                                                       tensor_quantized);
    }

    torch::Tensor forward_quantized(torch::Tensor x) {
        // Input x is [N, T, C], contiguity optional
        x = x.contiguous();

        // If this is the first time the forward method is being applied, do some startup
        if (device_w_hh.empty()) {
            for (auto &rnn : {rnn1, rnn2, rnn3, rnn4, rnn5}) {
                auto const &params = rnn->named_parameters();
                rearrange_individual_weights(params["weight_hh_l0"]);
                rearrange_individual_weights(params["weight_ih_l0"]);
                rearrange_individual_weights(params["bias_ih_l0"]);
                auto [scale, quantized] = quantize_tensor(params["weight_hh_l0"].t());
                device_w_ih.push_back(params["weight_ih_l0"].transpose(0, 1).contiguous());
                device_w_hh.push_back(quantized.contiguous());
                device_bias.push_back(params["bias_ih_l0"]);
                device_scale.push_back(scale.contiguous());
            }
        }

        // chunk_size * batch_size can not be > 2**31 (2147483648).
        // For practical purposes this is currently always the case.
        int batch_size = x.size(0);
        int chunk_size = x.size(1);
        // TODO: get rid of chunks buffer, as chunk size is fixed in Dorado
        auto _chunks = torch::empty({batch_size, 4}, torch::kInt32);
        _chunks.index({torch::indexing::Slice(), 0}) =
                torch::arange(0, chunk_size * batch_size, chunk_size);
        _chunks.index({torch::indexing::Slice(), 2}) =
                torch::arange(0, chunk_size * batch_size, chunk_size);
        _chunks.index({torch::indexing::Slice(), 1}) = chunk_size;
        _chunks.index({torch::indexing::Slice(), 3}) = 0;
        _chunks = _chunks.to(x.device());

        auto host_run_lstm_fwd =
                (layer_size == 96) ? host_run_lstm_fwd_quantized96 : host_run_lstm_fwd_quantized128;
        auto host_run_lstm_rev = (layer_size == 96) ? host_run_lstm_reverse_quantized96
                                                    : host_run_lstm_reverse_quantized128;

        auto buffer = torch::matmul(x, device_w_ih[0]);

        dorado::utils::handle_cuda_result(host_run_lstm_rev(
                _chunks.data_ptr(), buffer.data_ptr(), device_w_hh[0].data_ptr(),
                device_bias[0].data_ptr(), device_scale[0].data_ptr(), x.data_ptr(), x.size(0)));

        buffer = torch::matmul(x, device_w_ih[1]);

        dorado::utils::handle_cuda_result(host_run_lstm_fwd(
                _chunks.data_ptr(), buffer.data_ptr(), device_w_hh[1].data_ptr(),
                device_bias[1].data_ptr(), device_scale[1].data_ptr(), x.data_ptr(), x.size(0)));

        buffer = torch::matmul(x, device_w_ih[2]);

        dorado::utils::handle_cuda_result(host_run_lstm_rev(
                _chunks.data_ptr(), buffer.data_ptr(), device_w_hh[2].data_ptr(),
                device_bias[2].data_ptr(), device_scale[2].data_ptr(), x.data_ptr(), x.size(0)));

        buffer = torch::matmul(x, device_w_ih[3]);

        dorado::utils::handle_cuda_result(host_run_lstm_fwd(
                _chunks.data_ptr(), buffer.data_ptr(), device_w_hh[3].data_ptr(),
                device_bias[3].data_ptr(), device_scale[3].data_ptr(), x.data_ptr(), x.size(0)));

        buffer = torch::matmul(x, device_w_ih[4]);

        dorado::utils::handle_cuda_result(host_run_lstm_rev(
                _chunks.data_ptr(), buffer.data_ptr(), device_w_hh[4].data_ptr(),
                device_bias[4].data_ptr(), device_scale[4].data_ptr(), x.data_ptr(), x.size(0)));

        // Output is [N, T, C], contiguous
        return x;
    }

    std::vector<torch::Tensor> device_weights;
    std::vector<torch::Tensor> device_w_ih;
    std::vector<torch::Tensor> device_w_hh;
    std::vector<torch::Tensor> device_bias;
    std::vector<torch::Tensor> device_scale;
#endif  // if USE_CUDA_LSTM
    int layer_size;
    LSTM rnn1{nullptr}, rnn2{nullptr}, rnn3{nullptr}, rnn4{nullptr}, rnn5{nullptr};
};

struct ClampImpl : Module {
    ClampImpl(float _min, float _max, bool _active) : min(_min), max(_max), active(_active){};

    torch::Tensor forward(torch::Tensor x) {
        if (active) {
            x.clamp_(min, max);
        }
        return x;
    }

    bool active;
    float min, max;
};

TORCH_MODULE(LSTMStack);
TORCH_MODULE(LinearCRF);
TORCH_MODULE(Convolution);
TORCH_MODULE(Clamp);

struct CRFModelImpl : Module {
    CRFModelImpl(const CRFModelConfig &config, bool expand_blanks) {
        constexpr float conv_max_value = 3.5f;
        float conv3_max_value = g_options_conv3_max ? g_options_conv3_max : conv_max_value;
        conv1 = register_module("conv1", Convolution(config.num_features, config.conv, 5, 1,
                                                     config.clamp, conv_max_value, false));
        conv2 = register_module(
                "conv2", Convolution(config.conv, 16, 5, 1, config.clamp, conv_max_value, false));
        conv3 = register_module("conv3", Convolution(16, config.insize, 19, config.stride,
                                                     config.clamp, conv3_max_value, true));

        rnns = register_module("rnns", LSTMStack(config.insize));

        if (config.out_features.has_value()) {
            // The linear layer is decomposed into 2 matmuls.
            const int decomposition = config.out_features.value();
            linear1 = register_module("linear1", Linear(config.insize, decomposition));
            linear2 = register_module(
                    "linear2", Linear(LinearOptions(decomposition, config.outsize).bias(false)));
            clamp1 = Clamp(-5.0, 5.0, config.clamp);
            encoder = Sequential(conv1, conv2, conv3, rnns, linear1, linear2, clamp1);
        } else if ((config.conv == 16) && (config.num_features == 1)) {
            linear1 = register_module(
                    "linear1", Linear(LinearOptions(config.insize, config.outsize).bias(false)));
            clamp1 = Clamp(-5.0, 5.0, config.clamp);
            encoder = Sequential(conv1, conv2, conv3, rnns, linear1, clamp1);
        } else {
            linear = register_module("linear1", LinearCRF(config.insize, config.outsize));
            encoder = Sequential(conv1, conv2, conv3, rnns, linear);
        }
    }

    void load_state_dict(const std::vector<torch::Tensor> &weights) {
        utils::load_state_dict(*this, weights);
    }

    torch::Tensor forward(torch::Tensor x) {
        ScopedProfileRange spr("nn_forward");
        if (x.device() == torch::kCPU) {
            // Output is [T, N, C], which CPU decoding requires.
            return encoder->forward(x).transpose(0, 1);
        }
#if DORADO_GPU_BUILD && !defined(__APPLE__)
        try {
            // Output is [N, T, C]
            return encoder->forward(x);
        } catch (c10::Error &e) {
            spdlog::warn("Caught Torch error '{}', clearing CUDA cache and retrying.", e.msg());
            c10::cuda::CUDACachingAllocator::emptyCache();
        }
#endif
        // Output is [N, T, C]
        return encoder->forward(x);
    }

    LSTMStack rnns{nullptr};
    LinearCRF linear{nullptr};
    Linear linear1{nullptr}, linear2{nullptr};
    Sequential encoder{nullptr};
    Convolution conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    Clamp clamp1{nullptr};
};

TORCH_MODULE(CRFModel);

}  // namespace nn

CRFModelConfig load_crf_model_config(const std::filesystem::path &path) {
    const auto config_toml = toml::parse(path / "config.toml");

    CRFModelConfig config;
    config.model_path = path;

    if (config_toml.contains("qscore")) {
        const auto &qscore = toml::find(config_toml, "qscore");
        config.qbias = toml::find<float>(qscore, "bias");
        config.qscale = toml::find<float>(qscore, "scale");
        if (qscore.contains("mean_qscore_start_pos")) {
            config.mean_qscore_start_pos = toml::find<int32_t>(qscore, "mean_qscore_start_pos");
        }
    } else {
        spdlog::debug("> no qscore calibration found");
    }

    const auto &input = toml::find(config_toml, "input");
    config.num_features = toml::find<int>(input, "features");

    const auto &encoder = toml::find(config_toml, "encoder");
    if (encoder.contains("type")) {
        // v4-type model
        for (const auto &segment : toml::find(config_toml, "encoder", "sublayers").as_array()) {
            const auto type = toml::find<std::string>(segment, "type");
            if (type.compare("convolution") == 0) {
                // Overall stride is the product of all conv layers' strides.
                config.stride *= toml::find<int>(segment, "stride");
            } else if (type.compare("lstm") == 0) {
                config.insize = toml::find<int>(segment, "size");
            } else if (type.compare("linear") == 0) {
                // Specifying out_features implies a decomposition of the linear layer matrix
                // multiply with a bottleneck before the final feature size.
                config.out_features = toml::find<int>(segment, "out_features");
            } else if (type.compare("clamp") == 0) {
                config.clamp = true;
            } else if (type.compare("linearcrfencoder") == 0) {
                config.blank_score = toml::find<float>(segment, "blank_score");
            }
        }
        config.conv = 16;
        config.bias = config.insize > 128;
    } else {
        // pre-v4 model
        config.stride = toml::find<int>(encoder, "stride");
        config.insize = toml::find<int>(encoder, "features");
        config.blank_score = toml::find<float>(encoder, "blank_score");
        config.scale = toml::find<float>(encoder, "scale");

        if (encoder.contains("first_conv_size")) {
            config.conv = toml::find<int>(encoder, "first_conv_size");
        }
    }

    const auto &global_norm = toml::find(config_toml, "global_norm");
    // Note that in v4 files state_len appears twice: under global_norm and under
    // linearcrfencoder.  We are ignoring the latter.
    config.state_len = toml::find<int>(global_norm, "state_len");

    // All of the paths avoid outputting explicit stay scores from the NN,
    // so we have 4^bases * 4 transitions.
    const auto PowerOf4 = [](int x) { return 1 << (x << 1); };
    config.outsize = PowerOf4(config.state_len + 1);

    // Fetch run_info parameters.
    // Do nothing if run_info is not available in config file.
    if (config_toml.contains("run_info")) {
        const auto &run_info = toml::find(config_toml, "run_info");
        config.sample_rate = toml::find<int>(run_info, "sample_rate");
    }

    // Fetch signal normalisation parameters.
    // Use default values if normalisation section is not found.
    if (config_toml.contains("normalisation")) {
        const auto &norm = toml::find(config_toml, "normalisation");
        config.signal_norm_params.quantile_a = toml::find<float>(norm, "quantile_a");
        config.signal_norm_params.quantile_b = toml::find<float>(norm, "quantile_b");
        config.signal_norm_params.shift_multiplier = toml::find<float>(norm, "shift_multiplier");
        config.signal_norm_params.scale_multiplier = toml::find<float>(norm, "scale_multiplier");
    }

    // Set quantile scaling method based on the model filename
    std::string model_name = std::filesystem::canonical(config.model_path).filename().string();
    if (model_name.rfind("dna_r9.4.1", 0) == 0) {
        config.signal_norm_params.quantile_scaling = false;
    }

    return config;
}

std::vector<torch::Tensor> load_crf_model_weights(const std::filesystem::path &dir,
                                                  bool decomposition,
                                                  bool linear_layer_bias) {
    auto tensors = std::vector<std::string>{

            "0.conv.weight.tensor",      "0.conv.bias.tensor",

            "1.conv.weight.tensor",      "1.conv.bias.tensor",

            "2.conv.weight.tensor",      "2.conv.bias.tensor",

            "4.rnn.weight_ih_l0.tensor", "4.rnn.weight_hh_l0.tensor",
            "4.rnn.bias_ih_l0.tensor",   "4.rnn.bias_hh_l0.tensor",

            "5.rnn.weight_ih_l0.tensor", "5.rnn.weight_hh_l0.tensor",
            "5.rnn.bias_ih_l0.tensor",   "5.rnn.bias_hh_l0.tensor",

            "6.rnn.weight_ih_l0.tensor", "6.rnn.weight_hh_l0.tensor",
            "6.rnn.bias_ih_l0.tensor",   "6.rnn.bias_hh_l0.tensor",

            "7.rnn.weight_ih_l0.tensor", "7.rnn.weight_hh_l0.tensor",
            "7.rnn.bias_ih_l0.tensor",   "7.rnn.bias_hh_l0.tensor",

            "8.rnn.weight_ih_l0.tensor", "8.rnn.weight_hh_l0.tensor",
            "8.rnn.bias_ih_l0.tensor",   "8.rnn.bias_hh_l0.tensor",

            "9.linear.weight.tensor"};

    if (linear_layer_bias) {
        tensors.push_back("9.linear.bias.tensor");
    }

    if (decomposition) {
        tensors.push_back("10.linear.weight.tensor");
    }

    return utils::load_tensors(dir, tensors);
}

ModuleHolder<AnyModule> load_crf_model(const CRFModelConfig &model_config,
                                       const torch::TensorOptions &options) {
    const bool expand_blanks = !(USE_CUDA_LSTM && options.device().is_cuda());
    auto model = nn::CRFModel(model_config, expand_blanks);
    return populate_model(model, model_config.model_path, options,
                          model_config.out_features.has_value(), model_config.bias);
}

uint16_t get_model_sample_rate(const std::filesystem::path &model_path) {
    std::string model_name = std::filesystem::canonical(model_path).filename().string();
    // Find the sample rate from model config.
    int model_sample_rate = load_crf_model_config(model_path).sample_rate;
    if (model_sample_rate < 0) {
        // If unsuccessful, find sample rate by model name.
        model_sample_rate = utils::get_sample_rate_by_model_name(model_name);
    }
    return model_sample_rate;
}

int32_t get_model_mean_qscore_start_pos(const CRFModelConfig &model_config) {
    int32_t mean_qscore_start_pos = model_config.mean_qscore_start_pos;
    if (mean_qscore_start_pos < 0) {
        // If unsuccessful, find start position by model name.
        std::string model_name = model_config.model_path.filename().string();
        mean_qscore_start_pos = utils::get_mean_qscore_start_pos_by_model_name(model_name);
    }
    if (mean_qscore_start_pos < 0) {
        throw std::runtime_error("Mean q-score start position cannot be < 0");
    }
    return mean_qscore_start_pos;
}

}  // namespace dorado
