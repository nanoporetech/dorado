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

#define USE_KOI 1
#define CUDA_PROFILE_TO_CERR 0
#else
#define USE_KOI 0
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

#if USE_KOI

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

#endif  // if USE_KOI

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

#if 0
// This might come in handy for tracking down where big Torch allocations happen
void print_cuda_alloc_info(const std::string &label) {
    auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    auto print_stat_array = [](c10::cuda::CUDACachingAllocator::StatArray &stat,
                               const std::string &lbl) {
        constexpr float gig = 1024.f * 1024.f * 1024.f;
        std::cerr << lbl << "[" << stat[0].current / gig << ", " << stat[0].peak / gig << ", "
                  << stat[0].allocated / gig << ", " << stat[0].freed / gig << "] ";
    };
    std::cerr << "CUDAAlloc cpaf, " << label << " ";
    print_stat_array(stats.allocated_bytes, "All");
    print_stat_array(stats.reserved_bytes, "Rs");
    print_stat_array(stats.active_bytes, "Act");
    print_stat_array(stats.inactive_split_bytes, "In");
    std::cerr << std::endl;
}
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
static constexpr int WORKING_MEM_ALIGNMENT = 256;

/// Create a contiguous tensor view with `sizes` and `dtype`, using `working_mem` as backing memory.
/// if `from_front` is true, place at the front of `working_mem`, otherwise at the back.
torch::Tensor from_working_mem(torch::Tensor working_mem,
                               torch::IntArrayRef sizes,
                               torch::Dtype dtype,
                               bool from_front) {
    auto elems =
            std::accumulate(sizes.begin(), sizes.end(), int64_t(1), std::multiplies<int64_t>());
    auto elems_padded = utils::pad_to<int64_t>(elems, WORKING_MEM_ALIGNMENT);

    auto wm_dtype = working_mem.flatten().view(dtype);
    auto start_pos = from_front ? int64_t(0)
                                : ((wm_dtype.numel() - elems) / WORKING_MEM_ALIGNMENT) *
                                          WORKING_MEM_ALIGNMENT;
    return wm_dtype.slice(0, start_pos, start_pos + elems).view(sizes);
}

int64_t tensor_bytes(torch::IntArrayRef sizes, torch::Dtype dtype) {
    int64_t elems =
            std::accumulate(sizes.begin(), sizes.end(), int64_t(1), std::multiplies<int64_t>());
    return utils::pad_to<int64_t>(elems * torch::elementSize(dtype), WORKING_MEM_ALIGNMENT);
}

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

#if USE_KOI
    std::pair<int64_t, std::vector<int64_t>> get_working_mem_and_out_tensor_size(
            torch::IntArrayRef in_sizes) {
        int64_t batch_size = in_sizes[0];
        int64_t chunk_size_in = in_sizes[1];
        int64_t chunk_size_out = chunk_size_in / stride;
        int64_t in_buf_size = tensor_bytes(in_sizes, torch::kF16);
        if (to_lstm) {
            auto lstm_mode = get_cuda_lstm_mode(0, out_size);
            auto window_buf_size =
                    tensor_bytes({chunk_size_out, batch_size, in_size, window_size}, torch::kF16);
            if (lstm_mode == CUTLASS_TNC_I8) {
                std::vector<int64_t> out_sizes{chunk_size_out + 3, batch_size, out_size};
                auto mm_buf_size =
                        tensor_bytes({chunk_size_out, batch_size, out_size}, torch::kF16);
                auto out_buf_size = tensor_bytes(out_sizes, torch::kI8);
                auto working_buf_size =
                        std::max(window_buf_size + std::max(in_buf_size, mm_buf_size),
                                 mm_buf_size + out_buf_size);
                return std::make_pair(working_buf_size, out_sizes);
            } else {
                std::vector<int64_t> out_sizes;
                if (lstm_mode == QUANTISED_NTC) {
                    out_sizes = std::vector<int64_t>{chunk_size_out, batch_size, out_size};
                } else if (lstm_mode == CUTLASS_TNC_F16) {
                    out_sizes = std::vector<int64_t>{chunk_size_out + 3, batch_size, out_size};
                } else if (lstm_mode == CUBLAS_TN2C) {
                    out_sizes = std::vector<int64_t>{chunk_size_out + 1, batch_size, 2, out_size};
                } else {
                    throw std::logic_error("Unknown LSTM mode");
                }
                auto out_buf_size = tensor_bytes(out_sizes, torch::kF16);
                return std::make_pair(window_buf_size + std::max(in_buf_size, out_buf_size),
                                      out_sizes);
            }
        } else {
            std::vector<int64_t> out_sizes{batch_size, chunk_size_out, out_size};
            auto out_buf_size = tensor_bytes(out_sizes, torch::kF16);
            return std::make_pair(in_buf_size + out_buf_size, out_sizes);
        }
    }

    torch::Tensor run_koi(torch::Tensor working_mem, torch::Tensor in_view) {
        c10::cuda::CUDAGuard device_guard(in_view.device());
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        ScopedProfileRange spr("conv");

        bool in_is_front = (working_mem.data_ptr() == in_view.data_ptr());
        int batch_size = in_view.size(0);
        int chunk_size_in = in_view.size(1);
        int chunk_size_out = chunk_size_in / stride;

        // TODO: make device weights permanent, or use from_working_mem
        auto w_device = conv->weight.view({out_size, in_size * window_size})
                                .t()
                                .to(in_view.options())
                                .contiguous();
        auto b_device = conv->bias.to(in_view.options());

        if (to_lstm) {
            auto lstm_mode = get_cuda_lstm_mode(0, out_size);
            if (lstm_mode == QUANTISED_NTC) {
                auto ntcw_mat = from_working_mem(working_mem,
                                                 {batch_size, chunk_size_out, in_size, window_size},
                                                 torch::kF16, !in_is_front);
                auto res = from_working_mem(working_mem, {batch_size, chunk_size_out, out_size},
                                            torch::kF16, in_is_front);
                auto res_2D = res.view({-1, out_size});
                host_window_ntcw_f16(stream, in_view.stride(0), in_view.stride(1),
                                     in_view.stride(2), batch_size, chunk_size_in, in_size,
                                     window_size, stride, ntcw_mat.stride(0), ntcw_mat.stride(1),
                                     ntcw_mat.stride(2), ntcw_mat.stride(3), in_view.data_ptr(),
                                     ntcw_mat.data_ptr());
                dorado::utils::matmul_f16(ntcw_mat.view({-1, in_size * window_size}), w_device,
                                          res_2D);
                host_bias_swish_f16_clamp(stream, res_2D.size(0), res_2D.size(1), res_2D.stride(0),
                                          res_2D.data_ptr(), b_device.data_ptr(), max_value);
                // Output is [N, T_out, C_out], F16, contiguous
                return res;
            }
            auto tncw_mat = from_working_mem(working_mem,
                                             {chunk_size_out, batch_size, in_size, window_size},
                                             torch::kF16, !in_is_front);
            torch::Tensor res, mm_out;
            if (lstm_mode == CUTLASS_TNC_I8) {
                mm_out = from_working_mem(working_mem, {chunk_size_out * batch_size, out_size},
                                          torch::kF16, in_is_front);
                // Output is [T_out + 3, N, C_out], I8, contiguous
                res = from_working_mem(working_mem, {chunk_size_out + 3, batch_size, out_size},
                                       torch::kI8, !in_is_front);
            } else if (lstm_mode == CUTLASS_TNC_F16) {
                // Output is [T_out + 3, N, C_out], F16, contiguous
                res = from_working_mem(working_mem, {chunk_size_out + 3, batch_size, out_size},
                                       torch::kF16, in_is_front);
                mm_out = res.slice(0, 1, chunk_size_out + 1).view({-1, out_size});
            } else if (lstm_mode == CUBLAS_TN2C) {
                // Output is [T_out + 1, N, 2, C_out], F16, contiguous
                res = from_working_mem(working_mem, {chunk_size_out + 1, batch_size, 2, out_size},
                                       torch::kF16, in_is_front);
                auto res_TNC = res.slice(0, 1, chunk_size_out + 1).select(2, 1);
                mm_out = res_TNC.view({-1, out_size});
            }

            host_window_ntcw_f16(stream, in_view.stride(0), in_view.stride(1), in_view.stride(2),
                                 batch_size, chunk_size_in, in_size, window_size, stride,
                                 tncw_mat.stride(1), tncw_mat.stride(0), tncw_mat.stride(2),
                                 tncw_mat.stride(3), in_view.data_ptr(), tncw_mat.data_ptr());
            dorado::utils::matmul_f16(tncw_mat.view({-1, in_size * window_size}), w_device, mm_out);
            host_bias_swish_f16_clamp(stream, mm_out.size(0), mm_out.size(1), mm_out.stride(0),
                                      mm_out.data_ptr(), b_device.data_ptr(), max_value);

            if (lstm_mode == CUTLASS_TNC_I8) {
                auto out = res.slice(0, 1, chunk_size_out + 1).view({-1, out_size});
                host_convert(stream, mm_out.data_ptr(), 0, mm_out.stride(0), mm_out.stride(1),
                             KOI_F16, out.data_ptr(), 0, out.stride(0), out.stride(1), KOI_I8, 1,
                             out.size(0), out.size(1));
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
        } else {
            // Output is [N, T_out, C_out], contiguous
            auto res = from_working_mem(working_mem, {batch_size, chunk_size_out, out_size},
                                        torch::kF16, !in_is_front);
            if (host_convolution_swish_f16(stream, batch_size, in_size, out_size, chunk_size_in,
                                           window_size, stride, window_size / 2, in_view.data_ptr(),
                                           res.data_ptr(), w_device.data_ptr(), b_device.data_ptr(),
                                           max_value)) {
                return res;
            }
            throw std::runtime_error(std::string("Koi convolution failed with in size ") +
                                     std::to_string(in_size));
        }
    }
#endif

    torch::Tensor forward(torch::Tensor x) {
        // Input x is [N, C_in, T_in], contiguity optional
        ScopedProfileRange spr("conv");
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
#if USE_KOI
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
#endif  // if USE_KOI
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

struct LinearWrapperImpl : Module {
    LinearWrapperImpl(int insize, int outsize, bool bias_) : bias(bias_) {
        linear = register_module("linear", Linear(LinearOptions(insize, outsize).bias(bias)));
    };

    torch::Tensor forward(torch::Tensor x) {
        ScopedProfileRange spr("linear");
        // Input is [N, T, C], contiguity optional
        // Output is [N, T, C], contiguous
        return linear(x);
    }

#if USE_KOI
    std::pair<int64_t, std::vector<int64_t>> get_working_mem_and_out_tensor_size(
            torch::IntArrayRef in_sizes) {
        std::vector out_sizes{in_sizes[0], in_sizes[1], linear->weight.size(0)};
        auto in_bytes = tensor_bytes(in_sizes, torch::kF16);
        auto out_bytes = tensor_bytes(out_sizes, torch::kF16);
        return std::make_pair(in_bytes + out_bytes, out_sizes);
    }

    torch::Tensor run_koi(torch::Tensor working_mem, torch::Tensor in_view) {
        // Input is [N, T, C], contiguous
        c10::cuda::CUDAGuard device_guard(in_view.device());
        ScopedProfileRange spr("linear");

        if (wt.numel() == 0) {
            wt = linear->weight.t().contiguous();
        }
        bool in_is_front = (working_mem.data_ptr() == in_view.data_ptr());
        auto N = in_view.size(0);
        auto T = in_view.size(1);
        auto C = in_view.size(2);
        auto mm_out = from_working_mem(working_mem, {N * T, linear->weight.size(0)}, torch::kF16,
                                       !in_is_front);
        dorado::utils::matmul_f16(in_view.view({-1, C}), wt, mm_out);
        if (bias) {
            mm_out += linear->bias;
        }
        // Output is [N, T, C], contiguous
        return mm_out.view({N, T, -1});
    }

    torch::Tensor wt;
#endif  // if USE_KOI
    bool bias;
    Linear linear{nullptr};
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

    torch::Tensor forward(torch::Tensor x) {
        // Input is [N, T, C], contiguity optional

        auto [y1, h1] = rnn1(x.flip(1));
        auto [y2, h2] = rnn2(y1.flip(1));
        auto [y3, h3] = rnn3(y2.flip(1));
        auto [y4, h4] = rnn4(y3.flip(1));
        auto [y5, h5] = rnn5(y4.flip(1));

        // Output is [N, T, C], non-contiguous
        return y5.flip(1);
    }

#if USE_KOI
    std::pair<int64_t, std::vector<int64_t>> get_working_mem_and_out_tensor_size(
            torch::IntArrayRef in_sizes) {
        auto in_bytes_f16 = tensor_bytes(in_sizes, torch::kF16);
        auto mode = get_cuda_lstm_mode(0, layer_size);
        if (mode == CUTLASS_TNC_I8 || mode == CUTLASS_TNC_F16) {
            std::vector out_sizes{in_sizes[1], in_sizes[0] - 3, in_sizes[2]};
            auto buf_i8_bytes = tensor_bytes(in_sizes, torch::kI8);
            auto buf_f16_ntc_bytes = tensor_bytes(out_sizes, torch::kF16);
            auto mode_last = get_cuda_lstm_mode(4, layer_size);
            if (mode == CUTLASS_TNC_F16 && mode_last == CUTLASS_TNC_F16) {
                return std::make_pair(in_bytes_f16 + buf_f16_ntc_bytes, out_sizes);
            } else if (mode == CUTLASS_TNC_F16 && mode_last == CUTLASS_TNC_I8) {
                return std::make_pair(std::max(in_bytes_f16, buf_f16_ntc_bytes) + buf_i8_bytes,
                                      out_sizes);
            } else if (mode == CUTLASS_TNC_I8 && mode_last == CUTLASS_TNC_I8) {
                return std::make_pair(buf_f16_ntc_bytes + buf_i8_bytes, out_sizes);
            } else {
                throw std::logic_error("Invalid combination of Cutlass LSTM modes.");
            }
        } else if (mode == CUBLAS_TN2C) {
            std::vector out_sizes{in_sizes[1], in_sizes[0] - 1, in_sizes[3]};
            auto gate_bytes = tensor_bytes({in_sizes[1], 4 * layer_size}, torch::kF16);
            auto out_bytes = tensor_bytes(out_sizes, torch::kF16);
            return std::make_pair(in_bytes_f16 + std::max(gate_bytes, out_bytes), out_sizes);
        } else if (mode == QUANTISED_NTC) {
            auto gate_bytes =
                    tensor_bytes({in_sizes[0] * in_sizes[1], 4 * layer_size}, torch::kF16);
            return std::make_pair(in_bytes_f16 + gate_bytes, in_sizes.vec());
        }
        throw std::logic_error("Unknown LSTM mode");
    }

    torch::Tensor run_koi(torch::Tensor working_mem, torch::Tensor in_view) {
        c10::cuda::CUDAGuard device_guard(in_view.device());
        ScopedProfileRange spr("lstm_stack");

        // Output is [N, T, C], contiguous
        auto mode = get_cuda_lstm_mode(0, layer_size);
        if (mode == QUANTISED_NTC) {
            return forward_quantized(working_mem, in_view);
        } else if (mode == CUBLAS_TN2C) {
            return forward_cublas(working_mem, in_view);
        } else {
            return forward_cutlass(working_mem, in_view);
        }
    }

private:
    torch::Tensor forward_cublas(torch::Tensor working_mem, torch::Tensor in_view) {
        // input is ([T+1, N, 2, C], contiguous) (see below)
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        bool in_is_front = (working_mem.data_ptr() == in_view.data_ptr());
        const int chunk_size = in_view.size(0) - 1;
        const int batch_size = in_view.size(1);
        assert(layer_size == in_view.size(3));
        assert(in_view.dim() == 4 && in_view.size(2) == 2);
        assert(in_view.dtype() == torch::kF16);
        assert(in_view.is_contiguous());

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
        auto inout_all = in_view.view({chunk_size + 1, batch_size, -1});
        auto inout_left = in_view.slice(0, 0, chunk_size).select(2, 0);
        auto inout_right = in_view.slice(0, 1, chunk_size + 1).select(2, 1);

        auto gate_buf = from_working_mem(working_mem, {batch_size, layer_size * 4}, torch::kF16,
                                         !in_is_front);
        int layer_idx = 0;
        for (auto &rnn : {rnn1, rnn2, rnn3, rnn4, rnn5}) {
            bool reverse = !(layer_idx & 1);
            ScopedProfileRange spr_lstm("lstm_layer");
            auto state_buf = torch::zeros({batch_size, layer_size}, in_view.options());
            {
                if (int(device_weights.size()) == layer_idx) {
                    auto const &params = rnn->named_parameters();
                    auto w_ih = params["weight_ih_l0"];
                    auto w_hh = params["weight_hh_l0"];
                    device_bias.push_back(params["bias_ih_l0"].to(in_view.options()));
                    auto weights = torch::cat({reverse ? w_hh : w_ih, reverse ? w_ih : w_hh}, 1);
                    device_weights.push_back(weights.t().contiguous().to(in_view.options()));
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

        auto out = from_working_mem(working_mem, {batch_size, chunk_size, layer_size}, torch::kF16,
                                    !in_is_front);
        {
            ScopedProfileRange spr_convert("transpose_tn2c_to_ntc");
            auto &in = inout_left;
            host_transpose_f16(stream, in.data_ptr(), in.size(0), in.size(1), in.size(2),
                               in.stride(0), in.stride(1), in.stride(2), out.stride(1),
                               out.stride(0), out.stride(2), out.data_ptr());
        }
        // Output is [N, T, C], contiguous
        return out;
    }

    torch::Tensor forward_cutlass(torch::Tensor working_mem, torch::Tensor in_view) {
#ifdef DORADO_TX2  // Koi for TX2 does not have Cutlass kernels
        throw std::logic_error("No Cutlass kernels in Jetson TX2 build.");
#else
        // input is ([T+3, N, C], contiguous) (see below)
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        bool in_is_front = (working_mem.data_ptr() == in_view.data_ptr());
        const int chunk_size = in_view.size(0) - 3;
        const int batch_size = in_view.size(1);
        assert(layer_size == in_view.size(2));
        assert(in_view.dim() == 3);
        assert(in_view.dtype() == torch::kF16 || (use_int8 && in_view.dtype() == torch::kInt8));
        auto opts_f16 = in_view.options().dtype(torch::kF16);
        auto opts_i32 = in_view.options().dtype(torch::kI32);

        // Working memory is laid out as [T+3][N][C] in memory, where the reverse LSTM
        // layers (rnn1, rnn3, rnn5) use [1:-2] as input and [2:-1] as output, whereas the
        // forward LSTM layers (rnn2, rnn4) use [2:-1] as input and [1:-2] as output.
        // Note that both inout[0] and inout[-1] remain all zeroes, representing the initial
        // LSTM state h(-1) in either direction.

        torch::Tensor inout_all_f16, inout_all_i8, out_f16_NTC;
        int convert_to_int8_layer_idx = -1;
        if (in_view.dtype() == torch::kF16) {
            inout_all_f16 = in_view;
            convert_to_int8_layer_idx =
                    g_options_no_i8 ? 6 : 0;  // convert after first layer, or never
            if (g_options_no_i8) {
                out_f16_NTC = from_working_mem(working_mem, {batch_size, chunk_size, layer_size},
                                               torch::kF16, !in_is_front);
            } else {
                inout_all_i8 =
                        from_working_mem(working_mem, in_view.sizes(), torch::kI8, !in_is_front);
                out_f16_NTC = from_working_mem(working_mem, {batch_size, chunk_size, layer_size},
                                               torch::kF16, in_is_front);
            }
        } else if (in_view.dtype() == torch::kInt8) {
            inout_all_i8 = in_view;
            out_f16_NTC = from_working_mem(working_mem, {batch_size, chunk_size, layer_size},
                                           torch::kF16, !in_is_front);
        }

        // These illustrate the memory layout of the inputs to the fwd and reverse layers (not used)
        //    auto in_rev_i8 = inout_all_i8.slice(0, 1, chunk_size + 1);
        //    auto in_rev_f16 = inout_all_f16.slice(0, 1, chunk_size + 1);
        auto in_fwd_i8 = inout_all_i8.slice(0, 2, chunk_size + 2);
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
                device_weights.push_back(weights_cpu_cutlass.contiguous().to(in_view.device()));
            }

            auto in = (type_id == KOI_I8) ? inout_all_i8 : inout_all_f16;
            host_cutlass_lstm(stream, type_id, layer_idx, batch_size, layer_size, chunk_size,
                              reverse ? -1 : 1, in.stride(1), in.data_ptr(),
                              device_weights[layer_idx].data_ptr(),
                              device_bias[layer_idx].data_ptr(), device_scale[layer_idx].data_ptr(),
                              state_buf.data_ptr(), workspace_buf.data_ptr(), interleave, 0);

            if (layer_idx == convert_to_int8_layer_idx) {
                ScopedProfileRange spr_convert("f16_to_int8");
                auto &out = inout_all_i8;
                host_convert(stream, in.data_ptr(), in.stride(0), in.stride(1), in.stride(2),
                             KOI_F16, out.data_ptr(), out.stride(0), out.stride(1), out.stride(2),
                             KOI_I8, in.size(0), in.size(1), in.size(2));
            }

            ++layer_idx;
        }

        auto &out = out_f16_NTC;
        if (!g_options_no_i8) {
            ScopedProfileRange spr_convert("int8_tnc_to_f16_ntc");
            auto &in = in_fwd_i8;
            host_convert(stream, in.data_ptr(), in.stride(0), in.stride(1), in.stride(2), KOI_I8,
                         out.data_ptr(), out.stride(1), out.stride(0), out.stride(2), KOI_F16,
                         in.size(0), in.size(1), in.size(2));
        } else {
            ScopedProfileRange spr_convert("transpose_tnc_to_ntc");
            auto &in = in_fwd_f16;
            host_transpose_f16(stream, in.data_ptr(), in.size(0), in.size(1), in.size(2),
                               in.stride(0), in.stride(1), in.stride(2), out.stride(1),
                               out.stride(0), out.stride(2), out.data_ptr());
        }

        // Output is [N, T, C], contiguous
        return out;
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

    torch::Tensor forward_quantized(torch::Tensor working_mem, torch::Tensor in_view) {
        // Input is [N, T, C], contiguous
        int batch_size = in_view.size(0);
        int chunk_size = in_view.size(1);
        bool in_is_front = (working_mem.data_ptr() == in_view.data_ptr());

        // Quantise weights and move to GPU, if called for the first time
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
        // TODO: get rid of chunks buffer, as chunk size is fixed in Dorado
        auto chunks = torch::empty({batch_size, 4}, torch::kInt32);
        chunks.index({torch::indexing::Slice(), 0}) =
                torch::arange(0, chunk_size * batch_size, chunk_size);
        chunks.index({torch::indexing::Slice(), 2}) =
                torch::arange(0, chunk_size * batch_size, chunk_size);
        chunks.index({torch::indexing::Slice(), 1}) = chunk_size;
        chunks.index({torch::indexing::Slice(), 3}) = 0;
        chunks = chunks.to(in_view.device());

        auto host_run_lstm_fwd =
                (layer_size == 96) ? host_run_lstm_fwd_quantized96 : host_run_lstm_fwd_quantized128;
        auto host_run_lstm_rev = (layer_size == 96) ? host_run_lstm_reverse_quantized96
                                                    : host_run_lstm_reverse_quantized128;

        auto mm_out = from_working_mem(working_mem, {batch_size * chunk_size, 4 * layer_size},
                                       torch::kF16, !in_is_front);

        dorado::utils::matmul_f16(in_view.view({-1, layer_size}), device_w_ih[0], mm_out);
        dorado::utils::handle_cuda_result(
                host_run_lstm_rev(chunks.data_ptr(), mm_out.data_ptr(), device_w_hh[0].data_ptr(),
                                  device_bias[0].data_ptr(), device_scale[0].data_ptr(),
                                  in_view.data_ptr(), batch_size));

        dorado::utils::matmul_f16(in_view.view({-1, layer_size}), device_w_ih[1], mm_out);
        dorado::utils::handle_cuda_result(
                host_run_lstm_fwd(chunks.data_ptr(), mm_out.data_ptr(), device_w_hh[1].data_ptr(),
                                  device_bias[1].data_ptr(), device_scale[1].data_ptr(),
                                  in_view.data_ptr(), batch_size));

        dorado::utils::matmul_f16(in_view.view({-1, layer_size}), device_w_ih[2], mm_out);
        dorado::utils::handle_cuda_result(
                host_run_lstm_rev(chunks.data_ptr(), mm_out.data_ptr(), device_w_hh[2].data_ptr(),
                                  device_bias[2].data_ptr(), device_scale[2].data_ptr(),
                                  in_view.data_ptr(), batch_size));

        dorado::utils::matmul_f16(in_view.view({-1, layer_size}), device_w_ih[3], mm_out);
        dorado::utils::handle_cuda_result(
                host_run_lstm_fwd(chunks.data_ptr(), mm_out.data_ptr(), device_w_hh[3].data_ptr(),
                                  device_bias[3].data_ptr(), device_scale[3].data_ptr(),
                                  in_view.data_ptr(), batch_size));

        dorado::utils::matmul_f16(in_view.view({-1, layer_size}), device_w_ih[4], mm_out);
        dorado::utils::handle_cuda_result(
                host_run_lstm_rev(chunks.data_ptr(), mm_out.data_ptr(), device_w_hh[4].data_ptr(),
                                  device_bias[4].data_ptr(), device_scale[4].data_ptr(),
                                  in_view.data_ptr(), batch_size));

        // Output is [N, T, C], contiguous
        return in_view;
    }

    std::vector<torch::Tensor> device_weights;
    std::vector<torch::Tensor> device_w_ih;
    std::vector<torch::Tensor> device_w_hh;
    std::vector<torch::Tensor> device_bias;
    std::vector<torch::Tensor> device_scale;
#endif  // if USE_KOI
    int layer_size;
    LSTM rnn1{nullptr}, rnn2{nullptr}, rnn3{nullptr}, rnn4{nullptr}, rnn5{nullptr};
};

struct ClampImpl : Module {
    ClampImpl(float _min, float _max, bool _active) : min(_min), max(_max), active(_active){};

    torch::Tensor forward(torch::Tensor x) {
        ScopedProfileRange spr("clamp");
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
TORCH_MODULE(LinearWrapper);
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
            linear1 = register_module("linear1", LinearWrapper(config.insize, decomposition, true));
            linear2 =
                    register_module("linear2", LinearWrapper(decomposition, config.outsize, false));
            clamp1 = Clamp(-5.0, 5.0, config.clamp);
            encoder = Sequential(conv1, conv2, conv3, rnns, linear1, linear2, clamp1);
        } else if ((config.conv == 16) && (config.num_features == 1)) {
            linear1 =
                    register_module("linear1", LinearWrapper(config.insize, config.outsize, false));
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

#if USE_KOI
    torch::Tensor run_koi(torch::Tensor x) {
        // Input is [N, C, T]
        // TODO: change this on the input buffer side?
        x = x.transpose(1, 2).contiguous();

        // Determine working memory size in bytes
        auto [wb_size1, out_sizes1] = conv1->get_working_mem_and_out_tensor_size(x.sizes());
        auto [wb_size2, out_sizes2] = conv2->get_working_mem_and_out_tensor_size(out_sizes1);
        auto [wb_size3, out_sizes3] = conv3->get_working_mem_and_out_tensor_size(out_sizes2);
        auto [wb_size4, out_sizes4] = rnns->get_working_mem_and_out_tensor_size(out_sizes3);
        auto wb_size_bytes = std::max(std::max(wb_size1, wb_size2), std::max(wb_size3, wb_size4));
        if (linear1) {
            auto [wb_size5, out_sizes5] = linear1->get_working_mem_and_out_tensor_size(out_sizes4);
            wb_size_bytes = std::max(wb_size_bytes, wb_size5);
            if (linear2) {
                auto [wb_size6, out_sizes6] =
                        linear2->get_working_mem_and_out_tensor_size(out_sizes5);
                wb_size_bytes = std::max(wb_size_bytes, wb_size6);
            }
        }

        auto working_mem = torch::empty({wb_size_bytes}, x.options().dtype(torch::kI8));
        x = conv1->run_koi(working_mem, x);
        x = conv2->run_koi(working_mem, x);
        x = conv3->run_koi(working_mem, x);
        x = rnns->run_koi(working_mem, x);

        // Output is [N, T, C]
        if (linear1) {
            x = linear1->run_koi(working_mem, x);
            if (linear2) {
                x = linear2->run_koi(working_mem, x);
            }
            // TODO: clamp is entirely memory bandwidth bound, and relatively expensive due to the
            //  matrix size. Can we merge it into a custom linear kernel?
            x = clamp1->forward(x);
            return x;
        }
        // TODO: run_koi version of LinearCRF layer?
        return linear->forward(x);
    }
#endif

    torch::Tensor forward(torch::Tensor x) {
        ScopedProfileRange spr("nn_forward");
        if (x.device() == torch::kCPU) {
            // Output is [T, N, C], which CPU decoding requires.
            return encoder->forward(x).transpose(0, 1);
        }
#if USE_KOI
        // Output is [N, T, C]
        try {
            return run_koi(x);
        } catch (c10::Error &e) {
            spdlog::warn("Caught Torch error '{}', clearing CUDA cache and retrying.", e.msg());
            c10::cuda::CUDACachingAllocator::emptyCache();
        }
        return run_koi(x);
#else
        // Output is [N, T, C]
        return encoder->forward(x);
#endif
    }

    LSTMStack rnns{nullptr};
    LinearCRF linear{nullptr};
    LinearWrapper linear1{nullptr}, linear2{nullptr};
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
    const bool expand_blanks = !(USE_KOI && options.device().is_cuda());
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
