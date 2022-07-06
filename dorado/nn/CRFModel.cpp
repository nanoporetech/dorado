#include "CRFModel.h"

#include "../utils/tensor_utils.h"

#ifndef __APPLE__
#include <ATen/cuda/CUDAContext.h>
#endif

#include <math.h>
#include <toml.hpp>
#include <torch/torch.h>

using namespace torch::nn;
namespace F = torch::nn::functional;
using Slice = torch::indexing::Slice;

struct PermuteImpl : Module {
    torch::Tensor forward(torch::Tensor x) { return x.permute({2, 0, 1}); }
};

struct ConvolutionImpl : Module {
    ConvolutionImpl(int size = 1, int outsize = 1, int k = 1, int stride = 1) {
        conv = register_module(
                "conv", Conv1d(Conv1dOptions(size, outsize, k).stride(stride).padding(k / 2)));
        activation = register_module("activation", SiLU());
    }

    torch::Tensor forward(torch::Tensor x) { return activation(conv(x)); }

    Conv1d conv{nullptr};
    SiLU activation{nullptr};
};

struct LinearCRFImpl : Module {
    LinearCRFImpl(int insize, int outsize) : scale(5), blank_score(2.0), expand_blanks(false) {
        linear = register_module("linear", Linear(insize, outsize));
        activation = register_module("activation", Tanh());
    };

    torch::Tensor forward(torch::Tensor x) {
        auto scores = activation(linear(x)) * scale;

        if (expand_blanks == true) {
            int T = scores.size(0);
            int N = scores.size(1);
            int C = scores.size(2);
            scores = F::pad(scores.view({T, N, C / 4, 4}),
                            F::PadFuncOptions({1, 0, 0, 0, 0, 0, 0, 0}).value(blank_score))
                             .view({T, N, -1});
        }

        return scores;
    }

    int scale;
    int blank_score;
    bool expand_blanks;
    Linear linear{nullptr};
    Tanh activation{nullptr};
};

#define USE_CUDA_LSTM 1
#if USE_CUDA_LSTM
extern "C" {
#include "koi.h"
}
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(X)                                                                         \
    {                                                                                         \
        cudaError_t error = X;                                                                \
        if (error != cudaSuccess) {                                                           \
            printf("CUDA returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), \
                   error, __LINE__);                                                          \
            exit(EXIT_FAILURE);                                                               \
        }                                                                                     \
    }
#define CUBLAS_CHECK(X)                                                              \
    {                                                                                \
        cublasStatus_t res = X;                                                      \
        if (res != CUBLAS_STATUS_SUCCESS) {                                          \
            printf("CuBLAS returned error code %d, line(%d)\n", int(res), __LINE__); \
            exit(EXIT_FAILURE);                                                      \
        }                                                                            \
    }

static int get_cuda_device_id_from_device(const c10::Device &device) {
    if (!device.is_cuda() || !device.has_index()) {
        std::stringstream ss;
        ss << "Unable to extract CUDA device ID from device " << device;
        throw std::runtime_error(ss.str());
    }
    return device.index();
}

struct CudaLSTMImpl : Module {
    CudaLSTMImpl(int layer_size, bool reverse_) : reverse(reverse_) {
        // TODO: do we need to specify .device("gpu")?
        auto options = torch::TensorOptions().dtype(torch::kFloat16);
        weights = torch::empty({layer_size * 4, layer_size * 2}, options).contiguous();
        auto weight_ih = weights.slice(1, 0, layer_size);
        auto weight_hh = weights.slice(1, layer_size, 2 * layer_size);
        if (reverse) {
            std::swap(weight_ih, weight_hh);
        }
        bias = torch::empty({layer_size * 4}, options).contiguous();
        auto bias_hh = torch::empty({layer_size * 4}, options).contiguous();

        register_parameter("weight_ih", weight_ih, false);
        register_parameter("weight_hh", weight_hh, false);
        register_parameter("bias_ih", bias, false);
        register_parameter("bias_hh", bias_hh, false);
    }

    torch::Tensor weights, bias;
    bool reverse;
};

TORCH_MODULE(CudaLSTM);

struct LSTMStackImpl : Module {
    LSTMStackImpl(int layer_size_) : layer_size(layer_size_) {
        rnn1 = register_module("rnn_1", CudaLSTM(layer_size, true));
        rnn2 = register_module("rnn_2", CudaLSTM(layer_size, false));
        rnn3 = register_module("rnn_3", CudaLSTM(layer_size, true));
        rnn4 = register_module("rnn_4", CudaLSTM(layer_size, false));
        rnn5 = register_module("rnn_5", CudaLSTM(layer_size, true));
    }

    torch::Tensor forward(torch::Tensor in) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        auto tensor_options = torch::TensorOptions()
                                      .dtype(torch::kFloat16)
                                      .device(in.device())
                                      .requires_grad(false);
        int chunk_size = in.size(0);
        int batch_size = in.size(1);
        int gate_size = layer_size * 4;

        // We need some extra working memory to run the LSTM layers. By making it `thread_local`
        // this will work with multiple runners (i.e. multiple threads).
        thread_local torch::Tensor mat_working_mem, gate_buf, state_buf;
        thread_local int max_batch_size = 0;
        // TODO: even when not having to resize we need to make sure working mem buffers are on the right device
        if (batch_size > max_batch_size) {
            max_batch_size = batch_size;
            mat_working_mem =
                    torch::zeros({chunk_size + 1, batch_size, 2, layer_size}, tensor_options)
                            .contiguous()
                            .to(in.device());
            gate_buf = torch::empty({batch_size * gate_size}, tensor_options)
                               .contiguous()
                               .to(in.device());
            state_buf = torch::empty({batch_size * layer_size}, tensor_options)
                                .contiguous()
                                .to(in.device());
        }

        size_t timestep_buf_size = size_t(batch_size) * 2 * layer_size;
        int16_t *working_mem_ptr = (int16_t *)mat_working_mem.data_ptr();

        // copy contents of `in` into larger working memory matrix (holding both
        // input and output interleaved, as well as padding zeroes)

        // NOTE: `host_transpose_f16' does exactly what the commented out assignment
        // below would do, only ~5x faster (on A100)
        // mat_working_mem.slice(0, 1, chunk_size + 1).select(2, 1) = in;
        host_transpose_f16(stream, in.data_ptr(), in.size(0), in.size(1), in.size(2), in.stride(0),
                           in.stride(1), in.stride(2), batch_size * 2 * layer_size, 2 * layer_size,
                           1, working_mem_ptr + 2 * batch_size * layer_size + layer_size);

        for (auto &rnn : {rnn1, rnn2, rnn3, rnn4, rnn5}) {
            cudaMemsetAsync(state_buf.data_ptr(), 0,
                            size_t(batch_size) * layer_size * sizeof(int16_t), stream);
            // TODO: cache per-device weights and bias
            auto weights_cpu = rnn->weights.t().contiguous();
            auto weights = weights_cpu.to(in.device());
            auto bias = rnn->bias.to(in.device());
            for (int ts = 0; ts < chunk_size; ++ts) {
                void *timestep_in = working_mem_ptr + timestep_buf_size * ts;
                void *timestep_out = working_mem_ptr + timestep_buf_size * (ts + 1) + layer_size;
                if (rnn->reverse) {
                    timestep_out = working_mem_ptr + timestep_buf_size * (chunk_size - ts - 1);
                    timestep_in = working_mem_ptr + timestep_buf_size * (chunk_size - ts);
                }

                // timestep matrix mulitplication
                constexpr uint16_t HALF_ZERO = 0;      // 0.0 in __half format
                constexpr uint16_t HALF_ONE = 0x3C00;  // 1.0 in __half format
                CUBLAS_CHECK(cublasGemmEx(
                        at::cuda::getCurrentCUDABlasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, gate_size,
                        batch_size, 2 * layer_size, &HALF_ONE, (const void *)weights.data_ptr(),
                        CUDA_R_16F, gate_size, (const void *)timestep_in, CUDA_R_16F,
                        2 * layer_size, &HALF_ZERO, (void *)gate_buf.data_ptr(), CUDA_R_16F,
                        gate_size, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                host_lstm_step_f16(stream, batch_size, layer_size, bias.data_ptr(),
                                   gate_buf.data_ptr(), state_buf.data_ptr(), timestep_out);
            }
        }

        // TODO: can we modify and return `in` here, or define a `forward_inplace()` method?
        torch::Tensor out =
                torch::empty({in.size(1), in.size(0), in.size(2)}, tensor_options).transpose(0, 1);
        out.slice(0, 0, chunk_size) = mat_working_mem.index({Slice(0, chunk_size), Slice(), 0});
        return out;
    }

    int layer_size;
    CudaLSTM rnn1{nullptr}, rnn2{nullptr}, rnn3{nullptr}, rnn4{nullptr}, rnn5{nullptr};
};

#else  // if USE_CUDA_LSTM

struct LSTMStackImpl : Module {
    LSTMStackImpl(int size) {
        rnn1 = register_module("rnn1", LSTM(LSTMOptions(size, size)));
        rnn2 = register_module("rnn2", LSTM(LSTMOptions(size, size)));
        rnn3 = register_module("rnn3", LSTM(LSTMOptions(size, size)));
        rnn4 = register_module("rnn4", LSTM(LSTMOptions(size, size)));
        rnn5 = register_module("rnn5", LSTM(LSTMOptions(size, size)));
    };

    torch::Tensor forward(torch::Tensor x) {
        // rnn1
        x = x.flip(0);
        auto [y1, h1] = rnn1(x);
        x = y1.flip(0);

        // rnn2
        auto [y2, h2] = rnn2(x);
        x = y2;

        // rnn3
        x = x.flip(0);
        auto [y3, h3] = rnn3(x);
        x = y3.flip(0);

        // rnn4
        auto [y4, h4] = rnn4(x);
        x = y4;

        // rnn5
        x = x.flip(0);
        auto [y5, h5] = rnn5(x);
        x = y5.flip(0);

        return x;
    }

    LSTM rnn1{nullptr}, rnn2{nullptr}, rnn3{nullptr}, rnn4{nullptr}, rnn5{nullptr};
};

#endif  // if USE_CUDA_LSTM else

TORCH_MODULE(Permute);
TORCH_MODULE(LSTMStack);
TORCH_MODULE(LinearCRF);
TORCH_MODULE(Convolution);

struct CRFModelImpl : Module {
    CRFModelImpl(int size, int outsize, int stride, bool expand_blanks) {
        conv1 = register_module("conv1", Convolution(1, 4, 5, 1));
        conv2 = register_module("conv2", Convolution(4, 16, 5, 1));
        conv3 = register_module("conv3", Convolution(16, size, 19, stride));
        permute = register_module("permute", Permute());
        rnns = register_module("rnns", LSTMStack(size));
        linear = register_module("linear", LinearCRF(size, outsize));
        linear->expand_blanks = expand_blanks;
        encoder = Sequential(conv1, conv2, conv3, permute, rnns, linear);
    }

    void load_state_dict(std::vector<torch::Tensor> weights) {
        assert(weights.size() == parameters().size());
        for (size_t idx = 0; idx < weights.size(); idx++) {
            parameters()[idx].data() = weights[idx].data();
        }
    }

    torch::Tensor forward(torch::Tensor x) { return encoder->forward(x); }

    Permute permute{nullptr};
    LSTMStack rnns{nullptr};
    LinearCRF linear{nullptr};
    Sequential encoder{nullptr};
    Convolution conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
};

TORCH_MODULE(CRFModel);

ModuleHolder<AnyModule> load_crf_model(const std::string &path,
                                       int batch_size,
                                       int chunk_size,
                                       torch::TensorOptions options) {
    auto config = toml::parse(path + "/config.toml");

    const auto &encoder = toml::find(config, "encoder");
    const auto stride = toml::find<int>(encoder, "stride");
    const auto insize = toml::find<int>(encoder, "features");

    const auto &global_norm = toml::find(config, "global_norm");
    const auto state_len = toml::find<int>(global_norm, "state_len");
    int outsize = pow(4, state_len) * 4;
    bool expand = options.device_opt().value() == torch::kCPU;

    auto state_dict = load_weights(path);
    auto model = CRFModel(insize, outsize, stride, expand);
    model->load_state_dict(state_dict);
    model->to(options.dtype_opt().value().toScalarType());
    model->to(options.device_opt().value());
    model->eval();

    auto module = AnyModule(model);
    auto holder = ModuleHolder<AnyModule>(module);

    return holder;
}
