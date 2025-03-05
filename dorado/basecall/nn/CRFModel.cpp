#include "CRFModel.h"

#include "torch_utils/gpu_profiling.h"
#include "torch_utils/module_utils.h"
#include "torch_utils/tensor_utils.h"
#include "utils/math_utils.h"

#if DORADO_CUDA_BUILD
#include "torch_utils/cuda_utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

extern "C" {
#include "koi.h"
}
#endif

#include <torch/torch.h>

using namespace torch::nn;
namespace F = torch::nn::functional;
using Slice = torch::indexing::Slice;

namespace dorado::basecall::nn {

using namespace dorado::config;

#if DORADO_CUDA_BUILD
namespace {

KoiActivation get_koi_activation(Activation act) {
    if (act == Activation::SWISH) {
        return KOI_SWISH;
    } else if (act == Activation::SWISH_CLAMP) {
        return KOI_SWISH_CLAMP;
    } else if (act == Activation::TANH) {
        return KOI_TANH;
    } else {
        throw std::logic_error("Unrecognised activation function id.");
    }
}

// We have three different LSTM code paths:
//
// - Quantized: This path is only available for narrow LSTM layers, C == 96 or C == 128. It
//   uses CuBLAS GEMM (or torch::matmul) for the FP16 input-hidden matmul, and a custom kernel
//   using the DP4A instruction for the Int8*Int8->Int32 hidden-hidden matmul, and FP16 gate
//   computation. DP4A is not available on compute arch 6.2 (TX2).
//
// - Cutlass: This path is only available for LSTM layers where C is a multiple of 128 between
//   256 and 1024. It is currently only available on compute arch 8.0 (A100) and 9.0 (H100).
//   It uses a custom kernel based on the Cutlass library which performs Tensor Core matmul using
//   either F16 or Int8 and fuses the gate computation. FP16 is used only for the first LSTM layer,
//   and only if the output activation of the last convolution is not tanh.
// TODO: Add Cutlass kernels for 7.0 (V100, FP16) and for GPUs with less shared memory (7.x, 8.x)
//
// - CuBLAS: Slowest. This is the fallback path when none of the other paths applies. It uses
//   CuBLAS GEMM (or torch::matmul) plus `host_lstm_step_f16` from Koi. Uses FP16 precision.
//
// Each path needs its input in a different memory layout. To avoid extra transpose/conversion
// steps, the last convolution writes output in a memory layout suitable to serve as working memory
// for the first LSTM layer. (The specific memory layouts are further explained below in
// `LSTMStackImpl::forward_[cublas|cutlass]`.)
//
// These are the possible memory layouts for in/out buffers in working memory:
// [where T = chunk size ("time"), N = batch size, C = layer size ("channels")]
//
// - NTC: A contiguous tensor of size [N, T, C], dtype torch::kF16
// - TNC: A contiguous tensor of size [T, N, C], dtype torch::kF16
// - CUTLASS_TNC_F16: a contiguous tensor of size [T + 3, N, C], dtype torch::kF16
// - CUTLASS_TNC_I8: a contiguous tensor of size [T + 3, N, C], dtype torch::kI8
// - CUBLAS_TN2C: a contiguous tensor of size [T + 1, N, 2, C], dtype torch::kF16
//

// TODO: These should really be part of Koi
bool koi_can_use_cutlass() {
    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
    return (prop->major >= 8);
}
bool koi_can_use_quantised_lstm() {
    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
    // DP4A is supported on Pascal and later, except for TX2 (sm_62).
    return (prop->major > 6) || (prop->major == 6 && prop->minor != 2);
}

TensorLayout get_koi_lstm_input_layout(int layer_size, Activation activation) {
    TensorLayout layout = TensorLayout::CUBLAS_TN2C;
    if (koi_can_use_quantised_lstm() && (layer_size == 96 || layer_size == 128)) {
        layout = TensorLayout::NTC;
    } else if (koi_can_use_cutlass() && layer_size <= 1024 && layer_size > 128 &&
               (layer_size % 128) == 0) {
        layout = (activation == Activation::TANH) ? TensorLayout::CUTLASS_TNC_I8
                                                  : TensorLayout::CUTLASS_TNC_F16;
    }

    // Apply override (Cutlass override can only be applied if conditions are met)
    const char *env_lstm_mode = std::getenv("DORADO_LSTM_MODE");
    if (env_lstm_mode != nullptr) {
        std::string lstm_mode_str(env_lstm_mode);
        if (lstm_mode_str == "CUBLAS_TN2C") {
            layout = TensorLayout::CUBLAS_TN2C;
        } else if (lstm_mode_str == "CUTLASS_TNC_I8" && layout == TensorLayout::CUTLASS_TNC_F16) {
            layout = TensorLayout::CUTLASS_TNC_I8;
        } else if (lstm_mode_str == "CUTLASS_TNC_F16" && layout == TensorLayout::CUTLASS_TNC_I8) {
            layout = TensorLayout::CUTLASS_TNC_F16;
        }
    }

    return layout;
}

}  // namespace

// `WorkingMemory` encapsulates a backing tensor from which we create tensor views which map to
// either the front or the back of the backing tensor. The idea here is that we usually have one
// tensor with input data which we want to process to generate an output tensor. Once a processing
// step is done, the input tensor is no longer required and its memory can be reused, becoming the
// next output tensor. By creating views from alternating ends of one large tensor we can minimise
// the total amount of memory required.
//
// Sometimes the current tensor serves as both input and output of a processing step, but we also
// want a temporary buffer for the duration of the processing step. In that case `.temp()` can be
// called which creates a view of the specified size which will not be assigned to `.current`.
// A subsequent call to `.next_TC()` will create a view from the same end of the backing tensor,
// thus reusing the temp buffer memory.
//
// `.N`, the batch size, is constant for the lifetime of a `WorkingMemory` instance.
// `.T` (chunk size) and `.C` (channels) get updated with each call to `.next_TC()`
//
// Usage should be:
//   WorkingMemory wm(batch_size);
//   // Reservation phase, can mix `.next_TC()` and `.temp()`
//   wm.next_TC(chunk_size0, channels0, tensor_layout0);
//   wm.next_TC(chunk_size1, channels1, tensor_layout1);
//   wm.temp({dim0, ...}, dtype);
//   wm.next_TC(chunk_size2, channels2, tensor_layout2);
//    ...
//   wm.next_TC(chunk_sizeN, channelsN, tensor_layoutN);
//
//   // allocate_backing_tensor() begins use phase
//   wm.allocate_backing_tensor(device);
//
//   tensor0 = wm.next_TC(chunk_size0, channels0, tensor_layout0);
//    // write data to tensor0
//   tensor1 = wm.next_TC(chunk_size1, channels1, tensor_layout1);
//    // process: tensor0 -> tensor1
//   temp_tensor = wm.temp({dim0, ...}, dtype);
//    // process: tensor1 -> tensor1 with temp_tensor as temporary storage
//   tensor2 = wm.next_TC(chunk_size2, channels2, tensor_layout2);
//    // process: tensor1 -> tensor2
//    ...
//   tensorN = wm.next_TC(chunk_sizeN, channelsN, tensor_layoutN);
//    // process: tensorN-1 -> tensorN
//
// The pattern is: N calls to `.next_TC()/.temp()`, one call to `.allocate_backing_tensor()`,
// then N calls to `.next_TC()/.temp()` with the exact same parameters as before.

class WorkingMemory {
    // This may be overly conservative, but all CUDA allocation functions are guaranteed to
    // return 256-byte aligned pointers (even though GPU cache lines are at most 128 bytes).
    static constexpr int64_t ALIGNMENT = 256;

    static int64_t tensor_bytes(torch::IntArrayRef sizes, torch::Dtype dtype) {
        auto elems = c10::multiply_integers(sizes);
        return utils::pad_to<int64_t>(elems * torch::elementSize(dtype), ALIGNMENT);
    }

    at::Tensor next(torch::IntArrayRef sizes, torch::Dtype dtype, bool make_current) {
        auto new_bytes = tensor_bytes(sizes, dtype);
        at::Tensor new_tensor;
        if (!backing_tensor.defined()) {
            // If no backing tensor is allocated yet we're still in the reservation phase
            reservation_bytes = std::max(reservation_bytes, current_bytes + new_bytes);
        } else {
            if (current_bytes + new_bytes > reservation_bytes) {
                throw std::runtime_error("WorkingMemory: overlap detected.");
            }

            bool current_is_front =
                    current.defined() && current.data_ptr() == backing_tensor.data_ptr();
            auto elems = c10::multiply_integers(sizes);
            auto bt_dtype = backing_tensor.view(dtype);
            auto start_pos = current_is_front
                                     ? (reservation_bytes - new_bytes) / torch::elementSize(dtype)
                                     : int64_t(0);
            new_tensor = bt_dtype.narrow(0, start_pos, elems).view(sizes);
        }
        if (make_current) {
            current_bytes = new_bytes;
            current = new_tensor;
        }
        return new_tensor;
    }

public:
    explicit WorkingMemory(int batch_size) : N(batch_size) {}

    at::Tensor get_current_NTC_view() {
        switch (layout) {
        case TensorLayout::NTC:
            return current;
        case TensorLayout::TNC:
            return current.transpose(0, 1);
        case TensorLayout::CUTLASS_TNC_F16:
        case TensorLayout::CUTLASS_TNC_I8:
            return current.narrow(0, is_input_to_rev_lstm ? 1 : 2, T).transpose(1, 0);
        case TensorLayout::CUBLAS_TN2C:
            return current.narrow(0, is_input_to_rev_lstm ? 1 : 0, T)
                    .transpose(1, 0)
                    .select(2, is_input_to_rev_lstm ? 1 : 0);
        default:
            throw std::logic_error("Unhandled TensorLayout");
        }
    }

    at::Tensor next_TC(int T_, int C_, TensorLayout layout_) {
        T = T_;
        C = C_;
        layout = layout_;
        if (layout == TensorLayout::NTC) {
            return next({N, T, C}, torch::kF16, true);
        } else if (layout == TensorLayout::TNC) {
            return next({T, N, C}, torch::kF16, true);
        } else if (layout == TensorLayout::CUTLASS_TNC_F16) {
            return next({T + 3, N, C}, torch::kF16, true);
        } else if (layout == TensorLayout::CUTLASS_TNC_I8) {
            return next({T + 3, N, C}, torch::kI8, true);
        } else if (layout == TensorLayout::CUBLAS_TN2C) {
            return next({T + 1, N, 2, C}, torch::kF16, true);
        } else {
            throw std::logic_error("Unhandled TensorLayout");
        }
    }

    at::Tensor temp(torch::IntArrayRef sizes, torch::Dtype dtype) {
        return next(sizes, dtype, false);
    }

    void allocate_backing_tensor(torch::Device dev) {
        // Using kF16 here because the libtorch version on TX2 doesn't support `Tensor::view()`
        // with a dtype of a different size, and all buffers are kF16 on TX2.
        backing_tensor = torch::empty({reservation_bytes / 2},
                                      at::TensorOptions().device(dev).dtype(torch::kF16));
        current_bytes = 0;
    }

    int64_t reservation_bytes{0};
    int64_t current_bytes{0};
    at::Tensor backing_tensor;
    at::Tensor current;  // The last tensor view created with `next(_, _, true)`
    TensorLayout layout{TensorLayout::NTC};
    bool is_input_to_rev_lstm{true};
    const int N;  // batch size
    int T{0};     // current chunk size (time)
    int C{0};     // current layer size (channels)
};

#endif  // if DORADO_CUDA_BUILD

ConvStackImpl::ConvStackImpl(const std::vector<ConvParams> &layer_params) {
    for (size_t i = 0; i < layer_params.size(); ++i) {
        auto &layer = layers.emplace_back(layer_params[i]);
        auto opts = Conv1dOptions(layer.params.insize, layer.params.size, layer.params.winlen)
                            .stride(layer.params.stride)
                            .padding(layer.params.winlen / 2);
        layer.conv = register_module(std::string("conv") + std::to_string(i + 1), Conv1d(opts));
    }
}

#if DORADO_CUDA_BUILD
void ConvStackImpl::reserve_working_memory(WorkingMemory &wm) {
    auto &last = layers.back();
    last.output_layout = get_koi_lstm_input_layout(last.params.size, last.params.activation);
    last.cutlass_conv = utils::get_dev_opt<bool>("cutlass_conv", true) &&
                        (last.output_layout == TensorLayout::CUTLASS_TNC_I8 ||
                         last.output_layout == TensorLayout::CUTLASS_TNC_F16);
    if (last.cutlass_conv) {
        layers[layers.size() - 2].output_T_padding = last.params.winlen / 2;
    }
    for (auto &layer : layers) {
        layer.reserve_working_memory(wm);
    }
}

void ConvStackImpl::run_koi(WorkingMemory &wm) {
    for (auto &layer : layers) {
        layer.run_koi(wm);
    }
}
#endif  // if DORADO_CUDA_BUILD

at::Tensor ConvStackImpl::forward(at::Tensor x) {
    // Input x is [N, C_in, T_in], contiguity optional
    for (auto &layer : layers) {
        utils::ScopedProfileRange spr("conv", 2);
        x = layer.conv(x);
        if (layer.params.activation == Activation::SWISH) {
            torch::silu_(x);
        } else if (layer.params.activation == Activation::SWISH_CLAMP) {
            torch::silu_(x).clamp_(c10::nullopt, 3.5f);
        } else if (layer.params.activation == Activation::TANH) {
            x.tanh_();
        } else {
            throw std::logic_error("Unrecognised activation function id.");
        }
    }
    // Output is [N, T_out, C_out], non-contiguous
    return x.transpose(1, 2);
}

ConvStackImpl::ConvLayer::ConvLayer(const ConvParams &conv_params) : params(conv_params) {}

#if DORADO_CUDA_BUILD
void ConvStackImpl::ConvLayer::reserve_working_memory(WorkingMemory &wm) {
    assert(wm.layout == TensorLayout::NTC);
    const int T_in = wm.T;
    const int T_out = T_in / params.stride;
    const int C_in = wm.C;
    const int C_out = params.size;
    if (output_layout == TensorLayout::NTC && C_out > 16) {
        wm.next_TC(T_out, params.winlen * C_in, TensorLayout::NTC);
    } else if (cutlass_conv) {
    } else if (output_layout != TensorLayout::NTC) {
        wm.next_TC(T_out, params.winlen * C_in, TensorLayout::TNC);
        if (output_layout == TensorLayout::CUTLASS_TNC_I8) {
            wm.next_TC(T_out, C_out, TensorLayout::TNC);
        }
    }
    wm.next_TC(T_out + 2 * output_T_padding, C_out, output_layout);
}

void ConvStackImpl::ConvLayer::run_koi(WorkingMemory &wm) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    utils::ScopedProfileRange spr("conv", 2);

    auto in = wm.current;
    assert(wm.layout == TensorLayout::NTC);
    const int padding = (params.winlen / 2);
    const int T_in = cutlass_conv ? wm.T - 2 * padding : wm.T;
    const int T_out = T_in / params.stride;
    const int C_in = wm.C;
    const int C_out = params.size;

    if (!w_device.defined()) {
        auto opts = in.options().dtype(torch::kF16);
        // conv->weight is [C_out, C_in, W], we want [W, C_in, C_out]
        w_device = conv->weight.permute({2, 1, 0}).contiguous().flatten(0, 1).to(opts);
        if (cutlass_conv) {
            w_device = w_device.transpose(0, 1).contiguous();
        }
        b_device = conv->bias.to(opts);
    }

    if (output_layout == TensorLayout::NTC && C_out <= 16) {
        utils::ScopedProfileRange spr2("small conv", 3);
        auto out = wm.next_TC(T_out + 2 * output_T_padding, C_out, output_layout);
        out = out.narrow(1, output_T_padding, T_out);
        if (host_convolution_f16(stream, wm.N, C_in, C_out, T_in, params.winlen, params.stride,
                                 params.winlen / 2, int(out.stride(0)), in.data_ptr(),
                                 out.data_ptr(), w_device.data_ptr(), b_device.data_ptr(),
                                 get_koi_activation(params.activation))) {
            throw std::runtime_error(std::string("Koi convolution failed with in size ") +
                                     std::to_string(params.insize));
        }
    } else if (cutlass_conv) {
#if DORADO_TX2  // Koi for TX2 does not have Cutlass kernels
        throw std::logic_error("No Cutlass kernels in Jetson TX2 build.");
#else
        utils::ScopedProfileRange spr2("linear conv", 3);
        auto out_type = (output_layout == TensorLayout::CUTLASS_TNC_I8) ? KOI_I8 : KOI_F16;
        in.slice(1, 0, padding) = 0;
        in.slice(1, -padding, torch::indexing::None) = 0;
        wm.next_TC(T_out, C_out, output_layout);
        auto out_ntc = wm.get_current_NTC_view();
        auto res = host_linear(stream, KOI_F16, get_koi_activation(params.activation), out_type,
                               wm.N, T_out, C_in * params.winlen, C_out, int(in.stride(0)),
                               params.stride * C_in, int(out_ntc.stride(0)), int(out_ntc.stride(1)),
                               in.data_ptr(), w_device.data_ptr(), out_ntc.data_ptr(), nullptr,
                               b_device.data_ptr());
        if (res != KOI_SUCCESS) {
            throw std::runtime_error(std::string("Koi convolution failed with in size ") +
                                     std::to_string(params.insize));
        }
#endif  // DORADO_TX2
    } else {
        utils::ScopedProfileRange spr2("window conv", 3);
        // The window tensor is either NTC or TNC, depending on whether the first two
        // dimensions of the output layout are NT or TN.
        bool is_NT = (output_layout == TensorLayout::NTC);
        wm.next_TC(T_out, params.winlen * C_in, is_NT ? TensorLayout::NTC : TensorLayout::TNC);
        auto window_mat = wm.get_current_NTC_view();
        host_window_ntwc_f16(stream, wm.N, T_in, C_in, params.winlen, params.stride,
                             int(window_mat.stride(0)), int(window_mat.stride(1)), in.data_ptr(),
                             window_mat.data_ptr());

        auto mm_in = wm.current.flatten(0, 1);
        at::Tensor mm_out;
        if (output_layout == TensorLayout::NTC) {
            mm_out = wm.next_TC(T_out, C_out, output_layout);
        } else if (output_layout == TensorLayout::CUTLASS_TNC_I8) {
            mm_out = wm.next_TC(T_out, C_out, TensorLayout::TNC);
        } else {
            wm.next_TC(T_out, C_out, output_layout);
            mm_out = wm.get_current_NTC_view().transpose(0, 1);
        }

        mm_out = mm_out.view({-1, C_out});
        dorado::utils::matmul_f16(mm_in, w_device, mm_out);
        host_bias_activation_f16_inplace(
                stream, int(mm_out.size(0)), int(mm_out.size(1)), int(mm_out.stride(0)),
                mm_out.data_ptr(), b_device.data_ptr(), get_koi_activation(params.activation));

        if (output_layout == TensorLayout::CUTLASS_TNC_I8) {
            wm.next_TC(T_out, C_out, output_layout);
            auto conv_out = wm.get_current_NTC_view().transpose(0, 1).view({-1, C_out});
            host_convert(stream, mm_out.data_ptr(), 0, int(mm_out.stride(0)), int(mm_out.stride(1)),
                         KOI_F16, conv_out.data_ptr(), 0, int(conv_out.stride(0)),
                         int(conv_out.stride(1)), KOI_I8, 1, int(conv_out.size(0)),
                         int(conv_out.size(1)));
        }
    }
}
#endif  // if DORADO_CUDA_BUILD

LinearCRFImpl::LinearCRFImpl(int insize, int outsize, bool bias_, bool tanh_and_scale)
        : bias(bias_) {
    linear = register_module("linear", Linear(LinearOptions(insize, outsize).bias(bias)));
    if (tanh_and_scale) {
        activation = register_module("activation", Tanh());
    }
};

at::Tensor LinearCRFImpl::forward(const at::Tensor &x) {
    utils::ScopedProfileRange spr("linear", 2);
    // Input x is [N, T, C], contiguity optional
    auto scores = linear(x);
    if (activation) {
        scores = activation(scores) * scale;
    }

    // Output is [N, T, C], contiguous
    return scores;
}

#if DORADO_CUDA_BUILD
void LinearCRFImpl::reserve_working_memory(WorkingMemory &wm) {
    bool use_torch = utils::get_dev_opt<bool>("torch_linear", false) || !koi_can_use_cutlass();
    if (use_torch && wm.layout != TensorLayout::NTC) {
        wm.next_TC(wm.T, wm.C, TensorLayout::NTC);
    }
    wm.next_TC(wm.T, int(linear->weight.size(0)), TensorLayout::NTC);
}
void LinearCRFImpl::run_koi(WorkingMemory &wm) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto type_id = (wm.layout == TensorLayout::CUTLASS_TNC_I8) ? KOI_I8 : KOI_F16;
    int C_in = wm.C;
    int C_out = int(linear->weight.size(0));
    void *bias_ptr = bias ? linear->bias.data_ptr() : nullptr;

    bool use_torch = utils::get_dev_opt<bool>("torch_linear", false) || !koi_can_use_cutlass();
    if (use_torch || (wm.layout == TensorLayout::NTC && !activation)) {
        utils::ScopedProfileRange spr("linear", 2);
        if (wm.layout != TensorLayout::NTC) {
            // Convert/transpose input layout to NTC, F16 if necessary
            utils::ScopedProfileRange spr_convert("convert_to_f16_ntc", 3);
            auto in = wm.get_current_NTC_view();
            auto out = wm.next_TC(wm.T, wm.C, TensorLayout::NTC);
            host_convert(stream, in.data_ptr(), int(in.stride(0)), int(in.stride(1)),
                         int(in.stride(2)), type_id, out.data_ptr(), int(out.stride(0)),
                         int(out.stride(1)), int(out.stride(2)), KOI_F16, int(in.size(0)),
                         int(in.size(1)), int(in.size(2)));
        }

        auto in = wm.current;
        auto out = wm.next_TC(wm.T, C_out, TensorLayout::NTC);
        auto out_2D = out.view({-1, C_out});
        if (!w_device.defined()) {
            w_device = linear->weight.t().contiguous().to(in.device());
        }
        dorado::utils::matmul_f16(in.view({-1, C_in}), w_device, out_2D);
        if (activation) {
            host_bias_activation_f16_inplace(stream, wm.T * wm.N, C_out, C_out, out_2D.data_ptr(),
                                             bias_ptr, KOI_TANH_X5);
        } else if (bias) {
            out_2D += linear->bias;
        }
    } else {
#if DORADO_TX2  // Koi for TX2 does not have Cutlass kernels
        throw std::logic_error("No Cutlass kernels in Jetson TX2 build.");
#else
        utils::ScopedProfileRange spr("koi_linear", 2);
        auto in_ntc = wm.get_current_NTC_view();
        auto out = wm.next_TC(wm.T, C_out, TensorLayout::NTC);
        if (!w_device.defined()) {
            if (type_id == KOI_F16) {
                w_device = linear->weight.contiguous().to(in_ntc.options());
            } else {
                auto scaled_tensor = dorado::utils::quantize_tensor(linear->weight, 1);
                weight_scale = scaled_tensor.scale.to(torch::kF16).to(in_ntc.device());
                w_device = scaled_tensor.t.contiguous().to(in_ntc.device());
            }
        }
        auto res = host_linear(
                stream, type_id, activation ? KOI_TANH_X5 : KOI_IDENTITY, KOI_F16, wm.N, wm.T, C_in,
                C_out, int(in_ntc.stride(0)), int(in_ntc.stride(1)), int(out.stride(0)),
                int(out.stride(1)), in_ntc.data_ptr(), w_device.data_ptr(), out.data_ptr(),
                weight_scale.defined() ? weight_scale.data_ptr() : nullptr, bias_ptr);
        if (res != KOI_SUCCESS) {
            throw std::runtime_error(std::string("Linear layer error:") + std::to_string(res));
        }
#endif  // ifdef DORADO_TX2 else
    }
}
#endif

LSTMStackImpl::LSTMStackImpl(int num_layers, int size) : layer_size(size) {
    // torch::nn::LSTM expects/produces [N, T, C] with batch_first == true
    const auto lstm_opts = LSTMOptions(size, size).batch_first(true);
    for (int i = 0; i < num_layers; ++i) {
        auto label = std::string("rnn") + std::to_string(i + 1);
        rnns.emplace_back(register_module(label, LSTM(lstm_opts)));
    }
};

at::Tensor LSTMStackImpl::forward(at::Tensor x) {
    // Input is [N, T, C], contiguity optional
    for (auto &rnn : rnns) {
        x = std::get<0>(rnn(x.flip(1)));
    }

    // Output is [N, T, C], contiguous
    return (rnns.size() & 1) ? x.flip(1) : x;
}

#if DORADO_CUDA_BUILD
void LSTMStackImpl::reserve_working_memory(WorkingMemory &wm) {
    if (wm.layout == TensorLayout::NTC) {
        wm.temp({wm.N * wm.T, 4 * layer_size}, torch::kF16);
    } else if (wm.layout == TensorLayout::CUTLASS_TNC_F16) {
        wm.next_TC(wm.T, wm.C, TensorLayout::CUTLASS_TNC_I8);
    } else if (wm.layout == TensorLayout::CUBLAS_TN2C) {
        wm.temp({wm.N, 4 * layer_size}, torch::kF16);
    }
}

void LSTMStackImpl::run_koi(WorkingMemory &wm) {
    utils::ScopedProfileRange spr("lstm_stack", 2);

    if (wm.layout == TensorLayout::NTC) {
        return forward_quantized(wm);
    } else if (wm.layout == TensorLayout::CUBLAS_TN2C) {
        return forward_cublas(wm);
    } else if (wm.layout == TensorLayout::CUTLASS_TNC_F16 ||
               wm.layout == TensorLayout::CUTLASS_TNC_I8) {
        return forward_cutlass(wm);
    } else {
        throw std::runtime_error("Unhandled TensorLayout in LSTMStack.");
    }
}

void LSTMStackImpl::forward_cublas(WorkingMemory &wm) {
    // Working memory is laid out as [T+1][N][2][C] in memory, where the 2 serves to
    // interleave input and output for each LSTM layer in a specific way. The reverse LSTM
    // layers (even index) use right as input and left as output, whereas the forward
    // LSTM layers (odd index) use left as input and right as output.
    //
    // The interleaving means that x(t) and h(t-1), i.e. the input for the current timestep
    // and the output of the previous timestep, appear concatenated in memory and we can
    // perform a single matmul with the concatenated WU matrix
    // Note that both in[chunk_size][:][0][:] and in[0][:][1][:] remain
    // all zeroes, representing the initial LSTM state h(-1) in either direction.
    auto in = wm.current;
    in.index({0, Slice(), 1}) = 0;
    in.index({-1, Slice(), 0}) = 0;
    auto inout_all = in.flatten(2, 3);
    auto inout_left = in.narrow(0, 0, wm.T).select(2, 0);
    auto inout_right = in.narrow(0, 1, wm.T).select(2, 1);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto gate_buf = wm.temp({wm.N, layer_size * 4}, torch::kF16);

    for (size_t layer_idx = 0; layer_idx < rnns.size(); ++layer_idx) {
        bool reverse = !(layer_idx & 1);
        utils::ScopedProfileRange spr_lstm("lstm_layer", 3);
        auto state_buf = torch::zeros({wm.N, layer_size}, in.options());
        {
            // Move weights to GPU if called for the first time
            if (device_weights.size() == layer_idx) {
                const auto &params = rnns[layer_idx]->named_parameters();
                auto w_ih = params["weight_ih_l0"];
                auto w_hh = params["weight_hh_l0"];
                device_bias.push_back(params["bias_ih_l0"].to(in.options()));
                auto weights = torch::cat({reverse ? w_hh : w_ih, reverse ? w_ih : w_hh}, 1ll);
                device_weights.push_back(weights.t().contiguous().to(in.options()));
            }

            for (int ts = 0; ts < wm.T; ++ts) {
                auto timestep_in = inout_all[reverse ? (wm.T - ts) : ts];
                auto timestep_out = reverse ? inout_left[wm.T - ts - 1] : inout_right[ts];
                // Timestep matrix multiplication
                dorado::utils::matmul_f16(timestep_in, device_weights[layer_idx], gate_buf);
                host_lstm_step_f16(stream, wm.N, layer_size, device_bias[layer_idx].data_ptr(),
                                   gate_buf.data_ptr(), state_buf.data_ptr(),
                                   timestep_out.data_ptr());
            }
        }
        wm.is_input_to_rev_lstm = !reverse;
    }
}

void LSTMStackImpl::forward_cutlass(WorkingMemory &wm) {
#if DORADO_TX2  // Koi for TX2 does not have Cutlass kernels
    (void)wm;
    throw std::logic_error("No Cutlass kernels in Jetson TX2 build.");
#else
    // Working memory is laid out as [T+3][N][C] in memory, where the reverse LSTM
    // layers (even index) use [1:-2] as input and [2:-1] as output, whereas the
    // forward LSTM layers (odd index) use [2:-1] as input and [1:-2] as output.
    // Note that both inout[0] and inout[-1] remain all zeroes, representing the initial
    // LSTM state h(-1) in either direction.
    wm.current[0] = 0;
    wm.current[-1] = 0;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto opts_f16 = wm.current.options().dtype(torch::kF16);
    auto opts_i32 = opts_f16.dtype(torch::kI32);

    for (size_t layer_idx = 0; layer_idx < rnns.size(); ++layer_idx) {
        utils::ScopedProfileRange spr_lstm("lstm_layer", 3);
        bool reverse = !(layer_idx & 1);
        auto in = wm.current;
        auto type_id = (wm.layout == TensorLayout::CUTLASS_TNC_F16) ? KOI_F16 : KOI_I8;
        auto state_buf = torch::zeros({wm.N, layer_size}, opts_f16);
        auto workspace_buf = torch::empty({1024}, opts_i32);
        constexpr int interleave = 0;

        // Move weights to GPU if called for the first time
        if (device_weights.size() == layer_idx) {
            const auto &params = rnns[layer_idx]->named_parameters();
            // Both weight tensors are tensors of size  [4 * out_size, in_size],
            // where dimension 0 is Wi|Wf|Wg|Wo stacked, so it could more accurately be
            // described as [4, outsize, in_size]. Bias is alike, with the last dim dropped.
            auto w_ih = params["weight_ih_l0"].to(opts_f16);
            auto w_hh = params["weight_hh_l0"].to(opts_f16);
            auto weights_cpu = torch::cat({reverse ? w_hh : w_ih, reverse ? w_ih : w_hh}, 1);
            auto layer_device_bias = params["bias_ih_l0"].to(opts_f16).view({4, layer_size}).t();

            if (type_id == KOI_I8) {
                auto scaled_tensor = dorado::utils::quantize_tensor(weights_cpu, 1);
                weights_cpu = scaled_tensor.t;
                auto scale = scaled_tensor.scale.view({4, layer_size}).t();
                device_scale.push_back(scale.to(opts_f16).contiguous());
            } else {
                device_scale.push_back(torch::ones_like(layer_device_bias));
            }
            device_bias.push_back(layer_device_bias.contiguous());
            // Cutlass kernel expects weights reordered as <igigigigfofofofo>
            weights_cpu = weights_cpu.view({2, 2, -1, 4, 2 * layer_size});
            auto weights_cpu_cutlass =
                    weights_cpu.permute({2, 0, 3, 1, 4}).contiguous().view({-1, 2 * layer_size});
            if (interleave) {
                weights_cpu_cutlass = weights_cpu_cutlass.view({4 * layer_size, -1, interleave})
                                              .permute({1, 0, 2});
            }
            device_weights.push_back(weights_cpu_cutlass.contiguous().to(in.device()));
        }

        host_cutlass_lstm(stream, type_id, int(layer_idx), wm.N, layer_size, wm.T, reverse ? -1 : 1,
                          int(in.stride(1)), in.data_ptr(), device_weights[layer_idx].data_ptr(),
                          device_bias[layer_idx].data_ptr(), device_scale[layer_idx].data_ptr(),
                          state_buf.data_ptr(), workspace_buf.data_ptr(), interleave, 0);

        if (type_id == KOI_F16) {
            utils::ScopedProfileRange spr_convert("f16_to_int8", 4);
            auto out = wm.next_TC(wm.T, wm.C, TensorLayout::CUTLASS_TNC_I8);
            host_convert(stream, in.data_ptr(), int(in.stride(0)), int(in.stride(1)),
                         int(in.stride(2)), KOI_F16, out.data_ptr(), int(out.stride(0)),
                         int(out.stride(1)), int(out.stride(2)), KOI_I8, int(in.size(0)),
                         int(in.size(1)), int(in.size(2)));
        }

        wm.is_input_to_rev_lstm = !reverse;
    }
#endif  // DORADO_TX2
}

void LSTMStackImpl::forward_quantized(WorkingMemory &wm) {
    // Input and output in the same buffer, TensorLayout::NTC
    auto inout = wm.current;

    // Quantise weights and move to GPU, if called for the first time
    if (device_w_hh.empty()) {
        for (auto &rnn : rnns) {
            const auto &params = rnn->named_parameters();
            auto scaled_tensor = dorado::utils::quantize_tensor(params["weight_hh_l0"], 1);
            device_w_ih.push_back(params["weight_ih_l0"].transpose(0, 1).contiguous());
            device_w_hh.push_back(scaled_tensor.t.t().contiguous());
            device_bias.push_back(params["bias_ih_l0"]);
            device_scale.push_back(scaled_tensor.scale.contiguous());
        }
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto mm_out = wm.temp({wm.N * wm.T, 4 * layer_size}, torch::kF16);
    for (size_t i = 0; i < rnns.size(); ++i) {
        int dir = (i & 1) ? 1 : -1;
        dorado::utils::matmul_f16(inout.view({-1, layer_size}), device_w_ih[i], mm_out);
        dorado::utils::handle_cuda_result(host_small_lstm(
                stream, wm.N, wm.T, layer_size, dir, mm_out.data_ptr(), device_w_hh[i].data_ptr(),
                device_bias[i].data_ptr(), device_scale[i].data_ptr(), inout.data_ptr()));
    }
}
#endif  // if DORADO_CUDA_BUILD

ClampImpl::ClampImpl(float _min, float _max, bool _active)
        : active(_active), min(_min), max(_max) {}

at::Tensor ClampImpl::forward(at::Tensor x) {
    if (active) {
        utils::ScopedProfileRange spr("clamp", 2);
        x.clamp_(min, max);
    }
    return x;
}

CRFModelImpl::CRFModelImpl(const BasecallModelConfig &config) {
    const auto cv = config.convs;
    const auto lstm_size = config.lstm_size;
    convs = register_module("convs", ConvStack(cv));
    rnns = register_module("rnns", LSTMStack(5, lstm_size));

    if (config.out_features.has_value()) {
        // The linear layer is decomposed into 2 matmuls.
        const int decomposition = config.out_features.value();
        linear1 = register_module("linear1", LinearCRF(lstm_size, decomposition, true, false));
        linear2 =
                register_module("linear2", LinearCRF(decomposition, config.outsize, false, false));
        clamp1 = Clamp(-5.0, 5.0, config.clamp);
        encoder = Sequential(convs, rnns, linear1, linear2, clamp1);
    } else if ((config.convs[0].size > 4) && (config.num_features == 1)) {
        // v4.x model without linear decomposition
        linear1 = register_module("linear1", LinearCRF(lstm_size, config.outsize, false, false));
        clamp1 = Clamp(-5.0, 5.0, config.clamp);
        encoder = Sequential(convs, rnns, linear1, clamp1);
    } else {
        // Pre v4 model
        linear1 = register_module("linear1", LinearCRF(lstm_size, config.outsize, true, true));
        encoder = Sequential(convs, rnns, linear1);
    }
}

void CRFModelImpl::load_state_dict(const std::vector<at::Tensor> &weights) {
    utils::load_state_dict(*this, weights);
}

#if DORADO_CUDA_BUILD
at::Tensor CRFModelImpl::run_koi(const at::Tensor &in) {
    // Input is [N, C, T] -- TODO: change to [N, T, C] on the input buffer side?
    c10::cuda::CUDAGuard device_guard(in.device());

    // Determine working memory size
    WorkingMemory wm(int(in.size(0)));
    wm.next_TC(int(in.size(2)), int(in.size(1)), TensorLayout::NTC);
    convs->reserve_working_memory(wm);
    rnns->reserve_working_memory(wm);
    linear1->reserve_working_memory(wm);
    if (linear2) {
        linear2->reserve_working_memory(wm);
    }

    wm.allocate_backing_tensor(in.device());

    // Copy `in` to working memory and run the model
    auto wm_in = wm.next_TC(int(in.size(2)), int(in.size(1)), TensorLayout::NTC);
    wm_in.index({Slice()}) = in.transpose(1, 2);

    convs->run_koi(wm);
    rnns->run_koi(wm);
    linear1->run_koi(wm);
    if (linear2) {
        linear2->run_koi(wm);
    }

    // Clamping the scores to [-5, 5], if active (i.e. the role of `clamp1`), is performed by
    // `CUDADecoder` on reading the scores. This eliminates the cost of a large matrix
    // read-modify-write operation.

    // Output is [N, T, C], F16, contiguous
    assert(wm.layout == TensorLayout::NTC);
    return wm.current;
}
#endif

at::Tensor CRFModelImpl::forward(const at::Tensor &x) {
    utils::ScopedProfileRange spr("nn_forward", 1);
#if DORADO_CUDA_BUILD
    if (x.is_cuda() && x.dtype() == torch::kF16) {
        // Output is [N, T, C]
        return run_koi(x);
    }
#endif
    // Output is [N, T, C]
    return encoder->forward(x);
}

}  // namespace dorado::basecall::nn
