#include "CRFModel.h"

#include "CRFModelConfig.h"
#include "utils/gpu_profiling.h"
#include "utils/math_utils.h"
#include "utils/memory_utils.h"
#include "utils/module_utils.h"
#include "utils/tensor_utils.h"

#if DORADO_GPU_BUILD && !defined(__APPLE__)
#define USE_KOI 1

#include "../utils/cuda_utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

extern "C" {
#include "koi.h"
}

#else  // DORADO_GPU_BUILD && !defined(__APPLE__)
#define USE_KOI 0
#endif

#include <math.h>
#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <numeric>
#include <string>
#include <utility>

using namespace torch::nn;
namespace F = torch::nn::functional;
using Slice = torch::indexing::Slice;

#if USE_KOI

// We have four different LSTM code paths. They affect operation of both the LSTM layers and the
// last convolution layer. To avoid an extra transpose/copy, the last convolution writes output in
// a memory layout suitable to serve as working memory for the first LSTM layer. (The specific
// memory layouts are further explained below in `LSTMStackImpl::forward_[cublas|cutlass]`.)
//
// [Below, T = chunk size ("time", after conv3 stride), N = batch size, C = layer size ("channels")]
//
// - CUBLAS_TN2C: Slowest. This is the fallback mode when none of the other modes applies. It uses
//   CuBLAS GEMM (or torch::matmul) plus `host_lstm_step_f16` from Koi. Uses FP16 precision.
//   * LSTM input is a tensor of size [T + 1, N, 2, C], dtype torch::kF16
//
// - QUANTISED_NTC: This mode is only available for narrow LSTM layers, C == 96 or C == 128. It
//   uses CuBLAS GEMM (or torch::matmul) for the FP16 input-hidden matmul, and a custom kernel
//   using the DP4A instruction for the Int8*Int8->Int32 hidden-hidden matmul, and FP16 gate
//   computation. DP4A is not available on compute arch 6.2 (TX2).
//   * LSTM input is a tensor of size [N, T, C], dtype torch::kF16
//
// - CUTLASS_TNC_I8: This mode is only available for LSTM layers where C is a multiple of 128 and
//   C <= 1024. It is currently only available on compute arch 8.0 (A100) and 9.0 (H100). It uses
//   a custom kernel based on the Cutlass library for Int8*Int8->Int32 matmul and gate computation
//   producing Int8 output. Fastest mode. Used for all but the first LSTM layers.
//   * LSTM input is a tensor of size [T + 3, N, C], dtype torch::kI8
//
// - CUTLASS_TNC_F16: This mode is only available for LSTM layers where C is a multiple of 128 and
//   C <= 1024. It is currently only available on compute arch 8.0 (A100) and 9.0 (H100). It uses
//   a custom kernel based on the Cutlass library for FP16 matmul and gate computation. Used for
//   the first LSTM layer, as it needs to support inputs outside the [-1, 1] range.
//   * LSTM input is a tensor of size [T + 3, N, C], dtype torch::kF16
//
// TODO: Add Cutlass kernels for 7.0 (V100, FP16) and for GPUs with less shared memory (7.x, 8.x)

enum class LstmMode { CUBLAS_TN2C, QUANTISED_NTC, CUTLASS_TNC_I8, CUTLASS_TNC_F16 };

static LstmMode get_cuda_lstm_mode(int layer_idx, int layer_size) {
    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
    bool is_TX2 = (prop->major == 6 && prop->minor == 2);
    bool is_A100_H100 = ((prop->major == 8 || prop->major == 9) && prop->minor == 0);

    if (is_A100_H100 && layer_size <= 1024 && (layer_size % 128) == 0) {
        return (layer_idx == 0) ? LstmMode::CUTLASS_TNC_F16 : LstmMode::CUTLASS_TNC_I8;
    } else if (!is_TX2 && (layer_size == 96 || layer_size == 128)) {
        return LstmMode::QUANTISED_NTC;
    }
    return LstmMode::CUBLAS_TN2C;
}

// `WorkingMemory` encapsulates a backing tensor from which we create tensor views which map to
// either the front or the back of the backing tensor. The idea here is that we usually have one
// tensor with input data which we want to process to generate an output tensor. Once a processing
// step is done, the input tensor is no longer required and its memory can be reused, becoming the
// next output tensor. By creating views from alternating ends of one large tensor we can minimise
// the total amount of memory required.
//
// Sometimes the current tensor serves as both input and output of a processing step, but we also
// want a temporary buffer for the duration of the processing step. In that case `.next()` can be
// called with `make_current = false`, which means the newly created view will not be assigned to
// `.current`, and another call to `.next()` will create a view from the same end of the backing
// tensor, reusing the temp buffer memory.
//
// Usage should be:
//   WorkingMemory wm;
//   wm.reserve(t0_sizes, t0_dtype);
//    ...
//   wm.reserve(tN_sizes, tN_dtype);
//   wm.allocate_backing_tensor(device);
//   tensor0 = wm.next(t0_sizes, t0_dtype);
//    // write data to tensor0
//   tensor1 = wm.next(t1_sizes, t1_dtype);
//    // process: tensor0 -> tensor1
//    ...
//   tensorN = wm.next(tN_sizes, tN_dtype);
//    // process: tensorN-1 -> tensorN
//
// The pattern is: N calls to `.reserve()`, one call to `.allocate_backing_tensor()`, then N calls
// to `.next()` with the exact same parameters as the calls to `.reserve()`.

class WorkingMemory {
    // This may be overly conservative, but all CUDA allocation functions are guaranteed to
    // return 256-byte aligned pointers (even though GPU cache lines are at most 128 bytes).
    static constexpr int64_t ALIGNMENT = 256;

    int64_t tensor_bytes(torch::IntArrayRef sizes, torch::Dtype dtype) {
        auto elems = c10::multiply_integers(sizes);
        return dorado::utils::pad_to<int64_t>(elems * torch::elementSize(dtype), ALIGNMENT);
    }

public:
    torch::Tensor next(torch::IntArrayRef sizes, torch::Dtype dtype, bool make_current = true) {
        auto new_bytes = tensor_bytes(sizes, dtype);
        if ((current_bytes + new_bytes > reservation_bytes) ||
            backing_tensor.defined() && backing_tensor.nbytes() != reservation_bytes) {
            throw std::runtime_error("WorkingMemory: overlap detected.");
        }

        bool current_is_front =
                current.defined() && current.data_ptr() == backing_tensor.data_ptr();
        auto elems = c10::multiply_integers(sizes);
        auto bt_dtype = backing_tensor.view(dtype);
        auto start_pos = current_is_front
                                 ? (reservation_bytes - new_bytes) / torch::elementSize(dtype)
                                 : int64_t(0);
        auto new_tensor = bt_dtype.slice(0, start_pos, start_pos + elems).view(sizes);
        if (make_current) {
            current_bytes = new_bytes;
            current_sizes = sizes.vec();
            current = new_tensor;
        }
        return new_tensor;
    }

    void reserve(torch::IntArrayRef sizes, torch::Dtype dtype, bool make_current = true) {
        if (backing_tensor.defined()) {
            throw std::runtime_error("WorkingMemory: reserve after allocate.");
        }
        auto new_bytes = tensor_bytes(sizes, dtype);
        reservation_bytes = std::max(reservation_bytes, current_bytes + new_bytes);
        if (make_current) {
            current_bytes = new_bytes;
            current_sizes = sizes.vec();
        }
    }

    void allocate_backing_tensor(torch::Device dev) {
        // Using kF16 here because the libtorch version on TX2 doesn't support `Tensor::view()`
        // with a dtype of a different size, and all buffers are kF16 on TX2.
        backing_tensor = torch::empty({reservation_bytes / 2},
                                      torch::TensorOptions().device(dev).dtype(torch::kF16));
        current_sizes.clear();
        current_bytes = 0;
    }

    int64_t reservation_bytes{0};
    int64_t current_bytes{0};
    std::vector<int64_t> current_sizes;
    torch::Tensor backing_tensor;
    torch::Tensor current;  // The last tensor view created with `next(_, _, true)`
};

#endif  // if USE_KOI

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
                    bool next_layer_is_lstm_)
            : in_size(size),
              out_size(outsize),
              window_size(k),
              stride(stride_),
              clamp(clamp_),
              max_value(clamp_ ? max_value_ : std::numeric_limits<float>::max()),
              next_layer_is_lstm(next_layer_is_lstm_) {
        conv = register_module(
                "conv", Conv1d(Conv1dOptions(size, outsize, k).stride(stride).padding(k / 2)));
        activation = register_module("activation", SiLU());
    }

#if USE_KOI
    void reserve_working_memory(WorkingMemory &wm) {
        int64_t batch_size = wm.current_sizes[0];
        int64_t chunk_size_in = wm.current_sizes[1];
        int64_t chunk_size_out = chunk_size_in / stride;
        if (next_layer_is_lstm) {
            switch (auto lstm_mode = get_cuda_lstm_mode(0, out_size)) {
            case LstmMode::CUTLASS_TNC_I8:
                wm.reserve({chunk_size_out, batch_size, in_size, window_size}, torch::kF16);
                wm.reserve({chunk_size_out * batch_size, out_size}, torch::kF16);
                wm.reserve({chunk_size_out + 3, batch_size, out_size}, torch::kI8);
                break;
            case LstmMode::QUANTISED_NTC:
                wm.reserve({batch_size, chunk_size_out, in_size, window_size}, torch::kF16);
                wm.reserve({batch_size, chunk_size_out, out_size}, torch::kF16);
                break;
            case LstmMode::CUTLASS_TNC_F16:
                wm.reserve({chunk_size_out, batch_size, in_size, window_size}, torch::kF16);
                wm.reserve({chunk_size_out + 3, batch_size, out_size}, torch::kF16);
                break;
            case LstmMode::CUBLAS_TN2C:
                wm.reserve({chunk_size_out, batch_size, in_size, window_size}, torch::kF16);
                wm.reserve({chunk_size_out + 1, batch_size, 2, out_size}, torch::kF16);
                break;
            default:
                throw std::logic_error("Unknown LSTM mode");
            }
        } else {
            wm.reserve({batch_size, chunk_size_out, out_size}, torch::kF16);
        }
    }

    void run_koi(WorkingMemory &wm) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        utils::ScopedProfileRange spr("conv", 2);

        auto in = wm.current;
        int batch_size = in.size(0);
        int chunk_size_in = in.size(1);
        int chunk_size_out = chunk_size_in / stride;

        // TODO: make device weights permanent?
        auto w_device = conv->weight.view({out_size, in_size * window_size})
                                .t()
                                .to(in.options())
                                .contiguous();
        auto b_device = conv->bias.to(in.options());

        if (next_layer_is_lstm) {
            auto lstm_mode = get_cuda_lstm_mode(0, out_size);
            torch::Tensor ntcw_mat, tncw_mat;
            if (lstm_mode == LstmMode::QUANTISED_NTC) {
                ntcw_mat = wm.next({batch_size, chunk_size_out, in_size, window_size}, torch::kF16);
            } else {
                tncw_mat = wm.next({chunk_size_out, batch_size, in_size, window_size}, torch::kF16);
                ntcw_mat = tncw_mat.transpose(0, 1);
            }
            host_window_ntcw_f16(stream, in.stride(0), in.stride(1), in.stride(2), batch_size,
                                 chunk_size_in, in_size, window_size, stride, ntcw_mat.stride(0),
                                 ntcw_mat.stride(1), ntcw_mat.stride(2), ntcw_mat.stride(3),
                                 in.data_ptr(), ntcw_mat.data_ptr());

            auto mm_in = wm.current.view({-1, in_size * window_size});
            torch::Tensor mm_out, out;
            if (lstm_mode == LstmMode::QUANTISED_NTC) {
                // Output is [N, T_out, C_out], F16
                out = wm.next({batch_size, chunk_size_out, out_size}, torch::kF16);
                mm_out = out.view({-1, out_size});
            } else if (lstm_mode == LstmMode::CUTLASS_TNC_I8) {
                mm_out = wm.next({chunk_size_out * batch_size, out_size}, torch::kF16);
                out = wm.next({chunk_size_out + 3, batch_size, out_size}, torch::kI8);
                // Output is [T_out + 3, N, C_out], I8
            } else if (lstm_mode == LstmMode::CUTLASS_TNC_F16) {
                // Output is [T_out + 3, N, C_out], F16
                out = wm.next({chunk_size_out + 3, batch_size, out_size}, torch::kF16);
                mm_out = out.slice(0, 1, chunk_size_out + 1).view({-1, out_size});
            } else if (lstm_mode == LstmMode::CUBLAS_TN2C) {
                // Output is [T_out + 1, N, 2, C_out], F16
                out = wm.next({chunk_size_out + 1, batch_size, 2, out_size}, torch::kF16);
                auto out_TNC = out.slice(0, 1, chunk_size_out + 1).select(2, 1);
                mm_out = out_TNC.view({-1, out_size});
            }

            dorado::utils::matmul_f16(mm_in, w_device, mm_out);
            host_bias_swish_f16_clamp(stream, mm_out.size(0), mm_out.size(1), mm_out.stride(0),
                                      mm_out.data_ptr(), b_device.data_ptr(), max_value);

            if (lstm_mode == LstmMode::CUTLASS_TNC_I8) {
                auto conv_out = out.slice(0, 1, chunk_size_out + 1).view({-1, out_size});
                host_convert(stream, mm_out.data_ptr(), 0, mm_out.stride(0), mm_out.stride(1),
                             KOI_F16, conv_out.data_ptr(), 0, conv_out.stride(0),
                             conv_out.stride(1), KOI_I8, 1, conv_out.size(0), conv_out.size(1));
            }

            if (lstm_mode == LstmMode::CUBLAS_TN2C) {
                // Zero-fill the timesteps representing initial LSTM input (in both directions)
                out[0].select(1, 1) = 0;
                out[-1].select(1, 0) = 0;
            } else if (lstm_mode != LstmMode::QUANTISED_NTC) {
                out[0] = 0;
                out[-1] = 0;
            }
        } else {
            // Output is [N, T_out, C_out], contiguous
            auto out = wm.next({batch_size, chunk_size_out, out_size}, torch::kF16);
            if (host_convolution_swish_f16(stream, batch_size, in_size, out_size, chunk_size_in,
                                           window_size, stride, window_size / 2, in.data_ptr(),
                                           out.data_ptr(), w_device.data_ptr(), b_device.data_ptr(),
                                           max_value) != 0) {
                throw std::runtime_error(std::string("Koi convolution failed with in size ") +
                                         std::to_string(in_size));
            }
        }
    }
#endif

    torch::Tensor forward(torch::Tensor x) {
        // Input x is [N, C_in, T_in], contiguity optional
        utils::ScopedProfileRange spr("conv", 2);
        x = activation(conv(x));
        if (clamp) {
            x.clamp_(c10::nullopt, max_value);
        }
        if (next_layer_is_lstm) {
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
    const bool next_layer_is_lstm;
};

struct LinearCRFImpl : Module {
    LinearCRFImpl(int insize, int outsize, bool bias_, bool tanh_and_scale = false) : bias(bias_) {
        linear = register_module("linear", Linear(LinearOptions(insize, outsize).bias(bias)));
        if (tanh_and_scale) {
            activation = register_module("activation", Tanh());
        }
    };

    torch::Tensor forward(torch::Tensor x) {
        utils::ScopedProfileRange spr("linear", 2);
        // Input x is [N, T, C], contiguity optional
        auto scores = linear(x);
        if (activation) {
            scores = activation(scores) * scale;
        }

        // Output is [N, T, C], contiguous
        return scores;
    }

#if USE_KOI
    void reserve_working_memory(WorkingMemory &wm) {
        wm.reserve({wm.current_sizes[0], wm.current_sizes[1], linear->weight.size(0)}, torch::kF16);
    }
    void run_koi(WorkingMemory &wm) {
        utils::ScopedProfileRange spr("linear", 2);
        if (wt.numel() == 0) {
            wt = linear->weight.t().contiguous();
        }
        // Input is [N, T, Cin]
        auto in = wm.current;
        auto N = in.size(0);
        auto T = in.size(1);
        auto Cin = in.size(2);
        auto Cout = linear->weight.size(0);
        // Output is [N, T, Cout], contiguous
        auto out = wm.next({N, T, Cout}, torch::kF16);
        auto out_2D = out.view({-1, Cout});
        dorado::utils::matmul_f16(in.view({-1, Cin}), wt, out_2D);
        if (activation) {
            auto stream = at::cuda::getCurrentCUDAStream().stream();
            host_bias_tanh_scale_f16(stream, N * T, out_2D.size(1), scale, out_2D.data_ptr(),
                                     linear->bias.data_ptr());
        } else if (bias) {
            out_2D += linear->bias;
        }
    }

    torch::Tensor wt;
#endif  // if USE_KOI
    bool bias;
    static constexpr int scale = 5;
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
    void reserve_working_memory(WorkingMemory &wm) {
        auto in_sizes = wm.current_sizes;
        switch (auto mode = get_cuda_lstm_mode(0, layer_size)) {
        case LstmMode::CUTLASS_TNC_F16:
            if (get_cuda_lstm_mode(1, layer_size) == LstmMode::CUTLASS_TNC_I8) {
                wm.reserve(in_sizes, torch::kI8);
            }
            // fall-through
        case LstmMode::CUTLASS_TNC_I8:
            wm.reserve({in_sizes[1], in_sizes[0] - 3, layer_size}, torch::kF16);
            break;
        case LstmMode::CUBLAS_TN2C:
            wm.reserve({in_sizes[1], layer_size * 4}, torch::kF16, false);
            wm.reserve({in_sizes[1], in_sizes[0] - 1, layer_size}, torch::kF16);
            break;
        case LstmMode::QUANTISED_NTC:
            wm.reserve({in_sizes[0] * in_sizes[1], 4 * layer_size}, torch::kF16, false);
            break;
        default:
            throw std::logic_error("Unknown LSTM mode");
        }
    }

    void run_koi(WorkingMemory &wm) {
        utils::ScopedProfileRange spr("lstm_stack", 2);

        auto mode = get_cuda_lstm_mode(0, layer_size);
        if (mode == LstmMode::QUANTISED_NTC) {
            return forward_quantized(wm);
        } else if (mode == LstmMode::CUBLAS_TN2C) {
            return forward_cublas(wm);
        } else {
            return forward_cutlass(wm);
        }
        // Output is [N, T, C], F16
    }

private:
    void forward_cublas(WorkingMemory &wm) {
        // input is ([T+1, N, 2, C]) (see below)
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        auto in = wm.current;
        const int chunk_size = in.size(0) - 1;
        const int batch_size = in.size(1);
        assert(layer_size == in.size(3));
        assert(in.dim() == 4 && in.size(2) == 2);
        assert(in.dtype() == torch::kF16);

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

        auto gate_buf = wm.next({batch_size, layer_size * 4}, torch::kF16, false);
        int layer_idx = 0;
        for (auto &rnn : {rnn1, rnn2, rnn3, rnn4, rnn5}) {
            bool reverse = !(layer_idx & 1);
            utils::ScopedProfileRange spr_lstm("lstm_layer", 3);
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
                    // Timestep matrix multiplication
                    dorado::utils::matmul_f16(timestep_in, device_weights[layer_idx], gate_buf);
                    host_lstm_step_f16(stream, batch_size, layer_size,
                                       device_bias[layer_idx].data_ptr(), gate_buf.data_ptr(),
                                       state_buf.data_ptr(), timestep_out.data_ptr());
                }
            }
            ++layer_idx;
        }

        // Output is [N, T, C]
        auto out = wm.next({batch_size, chunk_size, layer_size}, torch::kF16);
        in = inout_left;
        utils::ScopedProfileRange spr_convert("transpose_tn2c_to_ntc", 3);
        host_transpose_f16(stream, in.data_ptr(), in.size(0), in.size(1), in.size(2), in.stride(0),
                           in.stride(1), in.stride(2), out.stride(1), out.stride(0), out.stride(2),
                           out.data_ptr());
    }

    void forward_cutlass(WorkingMemory &wm) {
#ifdef DORADO_TX2  // Koi for TX2 does not have Cutlass kernels
        throw std::logic_error("No Cutlass kernels in Jetson TX2 build.");
#else
        // input is [T+3, N, C] (see below)
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        auto in = wm.current;
        const int chunk_size = in.size(0) - 3;
        const int batch_size = in.size(1);
        assert(layer_size == in.size(2));
        assert(in.dim() == 3);
        assert(in.dtype() == torch::kF16 || in.dtype() == torch::kInt8);
        auto opts_f16 = in.options().dtype(torch::kF16);
        auto opts_i32 = in.options().dtype(torch::kI32);

        // Working memory is laid out as [T+3][N][C] in memory, where the reverse LSTM
        // layers (rnn1, rnn3, rnn5) use [1:-2] as input and [2:-1] as output, whereas the
        // forward LSTM layers (rnn2, rnn4) use [2:-1] as input and [1:-2] as output.
        // Note that both inout[0] and inout[-1] remain all zeroes, representing the initial
        // LSTM state h(-1) in either direction.

        auto type_id = (in.dtype() == torch::kF16) ? KOI_F16 : KOI_I8;
        bool convert_to_i8 = (type_id == KOI_F16) &&
                             (get_cuda_lstm_mode(1, layer_size) == LstmMode::CUTLASS_TNC_I8);

        int layer_idx = 0;
        for (auto &rnn : {rnn1, rnn2, rnn3, rnn4, rnn5}) {
            bool reverse = !(layer_idx & 1);
            utils::ScopedProfileRange spr_lstm("lstm_layer", 3);
            auto state_buf = torch::zeros({batch_size, layer_size}, opts_f16);
            auto workspace_buf = torch::empty({1024}, opts_i32);
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
                device_weights.push_back(weights_cpu_cutlass.contiguous().to(in.device()));
            }

            auto in = wm.current;
            host_cutlass_lstm(stream, type_id, layer_idx, batch_size, layer_size, chunk_size,
                              reverse ? -1 : 1, in.stride(1), in.data_ptr(),
                              device_weights[layer_idx].data_ptr(),
                              device_bias[layer_idx].data_ptr(), device_scale[layer_idx].data_ptr(),
                              state_buf.data_ptr(), workspace_buf.data_ptr(), interleave, 0);

            if (layer_idx == 0 && convert_to_i8) {
                utils::ScopedProfileRange spr_convert("f16_to_int8", 4);
                auto out = wm.next(in.sizes(), torch::kI8);
                host_convert(stream, in.data_ptr(), in.stride(0), in.stride(1), in.stride(2),
                             KOI_F16, out.data_ptr(), out.stride(0), out.stride(1), out.stride(2),
                             KOI_I8, in.size(0), in.size(1), in.size(2));
                type_id = KOI_I8;
            }

            ++layer_idx;
        }

        // Output is [N, T, C], F16
        in = wm.current.slice(0, 2, chunk_size + 2);
        auto out = wm.next({batch_size, chunk_size, layer_size}, torch::kF16);
        if (type_id == KOI_I8) {
            utils::ScopedProfileRange spr_convert("int8_tnc_to_f16_ntc", 3);
            host_convert(stream, in.data_ptr(), in.stride(0), in.stride(1), in.stride(2), KOI_I8,
                         out.data_ptr(), out.stride(1), out.stride(0), out.stride(2), KOI_F16,
                         in.size(0), in.size(1), in.size(2));
        } else {
            utils::ScopedProfileRange spr_convert("transpose_tnc_to_ntc", 3);
            host_transpose_f16(stream, in.data_ptr(), in.size(0), in.size(1), in.size(2),
                               in.stride(0), in.stride(1), in.stride(2), out.stride(1),
                               out.stride(0), out.stride(2), out.data_ptr());
        }

#endif  // ifdef DORADO_TX2 else
    }

    void rearrange_individual_weights(torch::Tensor buffer) {
        //Mapping of LSTM gate weights from IFGO to GIFO order.
        auto tmp = buffer.view({4, -1});
        tmp = torch::cat({tmp[2], tmp[0], tmp[1], tmp[3]});
        buffer.index({torch::indexing::Slice()}) = tmp.view(buffer.sizes());
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

        return {quantization_scale.to(torch::kFloat32), tensor_quantized};
    }

    void forward_quantized(WorkingMemory &wm) {
        // Input and output in the same buffer [N, T, C], F16
        auto inout = wm.current;
        int batch_size = inout.size(0);
        int chunk_size = inout.size(1);

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
        chunks = chunks.to(inout.device());

        auto host_run_lstm_fwd =
                (layer_size == 96) ? host_run_lstm_fwd_quantized96 : host_run_lstm_fwd_quantized128;
        auto host_run_lstm_rev = (layer_size == 96) ? host_run_lstm_reverse_quantized96
                                                    : host_run_lstm_reverse_quantized128;

        auto mm_out = wm.next({batch_size * chunk_size, 4 * layer_size}, torch::kF16, false);

        dorado::utils::matmul_f16(inout.view({-1, layer_size}), device_w_ih[0], mm_out);
        dorado::utils::handle_cuda_result(
                host_run_lstm_rev(chunks.data_ptr(), mm_out.data_ptr(), device_w_hh[0].data_ptr(),
                                  device_bias[0].data_ptr(), device_scale[0].data_ptr(),
                                  inout.data_ptr(), batch_size));

        dorado::utils::matmul_f16(inout.view({-1, layer_size}), device_w_ih[1], mm_out);
        dorado::utils::handle_cuda_result(
                host_run_lstm_fwd(chunks.data_ptr(), mm_out.data_ptr(), device_w_hh[1].data_ptr(),
                                  device_bias[1].data_ptr(), device_scale[1].data_ptr(),
                                  inout.data_ptr(), batch_size));

        dorado::utils::matmul_f16(inout.view({-1, layer_size}), device_w_ih[2], mm_out);
        dorado::utils::handle_cuda_result(
                host_run_lstm_rev(chunks.data_ptr(), mm_out.data_ptr(), device_w_hh[2].data_ptr(),
                                  device_bias[2].data_ptr(), device_scale[2].data_ptr(),
                                  inout.data_ptr(), batch_size));

        dorado::utils::matmul_f16(inout.view({-1, layer_size}), device_w_ih[3], mm_out);
        dorado::utils::handle_cuda_result(
                host_run_lstm_fwd(chunks.data_ptr(), mm_out.data_ptr(), device_w_hh[3].data_ptr(),
                                  device_bias[3].data_ptr(), device_scale[3].data_ptr(),
                                  inout.data_ptr(), batch_size));

        dorado::utils::matmul_f16(inout.view({-1, layer_size}), device_w_ih[4], mm_out);
        dorado::utils::handle_cuda_result(
                host_run_lstm_rev(chunks.data_ptr(), mm_out.data_ptr(), device_w_hh[4].data_ptr(),
                                  device_bias[4].data_ptr(), device_scale[4].data_ptr(),
                                  inout.data_ptr(), batch_size));
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
        if (active) {
            utils::ScopedProfileRange spr("clamp", 2);
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
    explicit CRFModelImpl(const CRFModelConfig &config) {
        constexpr float conv_max_value = 3.5f;
        conv1 = register_module("conv1", Convolution(config.num_features, config.conv, 5, 1,
                                                     config.clamp, conv_max_value, false));
        conv2 = register_module(
                "conv2", Convolution(config.conv, 16, 5, 1, config.clamp, conv_max_value, false));
        conv3 = register_module("conv3", Convolution(16, config.insize, 19, config.stride,
                                                     config.clamp, conv_max_value, true));

        rnns = register_module("rnns", LSTMStack(config.insize));

        if (config.out_features.has_value()) {
            // The linear layer is decomposed into 2 matmuls.
            const int decomposition = config.out_features.value();
            linear1 = register_module("linear1", LinearCRF(config.insize, decomposition, true));
            linear2 = register_module("linear2", LinearCRF(decomposition, config.outsize, false));
            clamp1 = Clamp(-5.0, 5.0, config.clamp);
            encoder = Sequential(conv1, conv2, conv3, rnns, linear1, linear2, clamp1);
        } else if ((config.conv == 16) && (config.num_features == 1)) {
            linear1 = register_module("linear1", LinearCRF(config.insize, config.outsize, false));
            clamp1 = Clamp(-5.0, 5.0, config.clamp);
            encoder = Sequential(conv1, conv2, conv3, rnns, linear1, clamp1);
        } else {
            linear1 = register_module("linear1",
                                      LinearCRF(config.insize, config.outsize, true, true));
            encoder = Sequential(conv1, conv2, conv3, rnns, linear1);
        }
    }

    void load_state_dict(const std::vector<torch::Tensor> &weights) {
        utils::load_state_dict(*this, weights);
    }

#if USE_KOI
    torch::Tensor run_koi(torch::Tensor in) {
        // Input is [N, C, T] -- TODO: change to [N, C, T] on the input buffer side?
        c10::cuda::CUDAGuard device_guard(in.device());

        WorkingMemory wm;
        wm.reserve(in.transpose(1, 2).sizes(), torch::kF16);
        // Determine working memory size
        conv1->reserve_working_memory(wm);
        conv2->reserve_working_memory(wm);
        conv3->reserve_working_memory(wm);
        rnns->reserve_working_memory(wm);
        linear1->reserve_working_memory(wm);
        if (linear2) {
            linear2->reserve_working_memory(wm);
        }

        wm.allocate_backing_tensor(in.device());

        auto stream = at::cuda::getCurrentCUDAStream().stream();
        auto out = wm.next(in.transpose(1, 2).sizes(), torch::kF16);
        host_transpose_f16(stream, in.data_ptr(), in.size(0), in.size(1), in.size(2), in.stride(0),
                           in.stride(1), in.stride(2), out.stride(0), out.stride(2), out.stride(1),
                           out.data_ptr());

        conv1->run_koi(wm);
        conv2->run_koi(wm);
        conv3->run_koi(wm);
        rnns->run_koi(wm);

        linear1->run_koi(wm);
        if (linear2) {
            linear2->run_koi(wm);
        }

        out = wm.current;
        if (clamp1) {
            // TODO: clamp is entirely memory bandwidth bound, and relatively expensive due to the
            //  matrix size. Can we merge it into a custom linear kernel?
            out = clamp1->forward(out);
        }
        // Output is [N, T, C], F16, contiguous
        return out;
    }
#endif

    torch::Tensor forward(torch::Tensor x) {
        utils::ScopedProfileRange spr("nn_forward", 1);
        if (x.device() == torch::kCPU) {
            // Output is [T, N, C], which CPU decoding requires.
            return encoder->forward(x).transpose(0, 1);
        }
#if USE_KOI
        if (x.is_cuda() && x.dtype() == torch::kF16) {
            // Output is [N, T, C]
            return run_koi(x);
        }
#endif
        // Output is [N, T, C]
        return encoder->forward(x);
    }

    LSTMStack rnns{nullptr};
    LinearCRF linear1{nullptr}, linear2{nullptr};
    Sequential encoder{nullptr};
    Convolution conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    Clamp clamp1{nullptr};
};

TORCH_MODULE(CRFModel);

}  // namespace nn

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
    auto model = nn::CRFModel(model_config);
    return populate_model(model, model_config.model_path, options,
                          model_config.out_features.has_value(), model_config.bias);
}

size_t auto_calculate_num_runners(const CRFModelConfig &model_config,
                                  size_t batch_size,
                                  float memory_fraction) {
    auto model_name = std::filesystem::canonical(model_config.model_path).filename().string();

    // very hand-wavy determination
    // these numbers were determined empirically by running 1, 2, 4 and 8 runners for each model
    auto required_ram_per_runner_GB = 0.f;
    if (model_name.find("_fast@v") != std::string::npos) {
        required_ram_per_runner_GB = 1.5;
    } else if (model_name.find("_hac@v") != std::string::npos) {
        required_ram_per_runner_GB = 4.5;
    } else if (model_name.find("_sup@v") != std::string::npos) {
        required_ram_per_runner_GB = 12.5;
    } else {
        return 1;
    }

    // numbers were determined with a batch_size of 128, assume this just scales
    required_ram_per_runner_GB *= batch_size / 128.f;

    auto free_ram_GB = utils::available_host_memory_GB() * memory_fraction;
    auto num_runners = static_cast<size_t>(free_ram_GB / required_ram_per_runner_GB);
    return std::clamp(num_runners, std::size_t(1),
                      std::size_t(std::thread::hardware_concurrency()));
}

}  // namespace dorado
