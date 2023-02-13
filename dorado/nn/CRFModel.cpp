#include "CRFModel.h"

#include "../utils/module_utils.h"
#include "../utils/tensor_utils.h"

#ifndef __APPLE__
#include "../utils/cuda_utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

extern "C" {
#include "koi.h"
}

#define USE_CUDA_LSTM 1
#else
#define USE_CUDA_LSTM 0
#endif

#include <math.h>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>
#include <toml.hpp>
#include <torch/torch.h>

#include <string>

using namespace torch::nn;
namespace F = torch::nn::functional;
using Slice = torch::indexing::Slice;
using quantized_lstm = std::function<int(void *, void *, void *, void *, void *, void *, int)>;

#if USE_CUDA_LSTM

static bool cuda_lstm_is_quantized(int layer_size) {
    return ((layer_size == 96) ||
            (layer_size ==
             128));  // TODO - change back! just a test to see if quantized kernels are the problem
}
#endif  // if USE_CUDA_LSTM

namespace {
template <class Model>
ModuleHolder<AnyModule> populate_model(Model &&model,
                                       const std::filesystem::path &path,
                                       const torch::TensorOptions &options,
                                       bool decomposition,
                                       bool bias) {
    auto state_dict = dorado::load_crf_model_weights(path, decomposition, bias);
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

struct ConvolutionImpl : Module {
    ConvolutionImpl(int size, int outsize, int k, int stride_, bool to_lstm_ = false)
            : in_size(size), out_size(outsize), window_size(k), stride(stride_), to_lstm(to_lstm_) {
        conv = register_module(
                "conv", Conv1d(Conv1dOptions(size, outsize, k).stride(stride).padding(k / 2)));
        activation = register_module("activation", SiLU());
    }

    torch::Tensor forward(torch::Tensor x) {
        // Input x is [N, C_in, T_in], contiguity optional
        if (to_lstm) {
#if USE_CUDA_LSTM
            if (x.device() != torch::kCPU) {
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
                if (cuda_lstm_is_quantized(out_size)) {
                    torch::Tensor res =
                            torch::empty({batch_size, chunk_size_out, out_size}, x.options());
                    auto res_2D = res.view({-1, out_size});
                    auto ntcw_mat = torch::empty({batch_size, chunk_size_out, in_size, window_size},
                                                 x.options());
                    host_window_ntcw_f16(stream, x.stride(0), x.stride(2), x.stride(1), batch_size,
                                         chunk_size_in, in_size, window_size, stride,
                                         ntcw_mat.stride(0), ntcw_mat.stride(1), ntcw_mat.stride(2),
                                         ntcw_mat.stride(3), x.data_ptr(), ntcw_mat.data_ptr());
                    dorado::utils::matmul_f16(ntcw_mat.view({-1, in_size * window_size}), w_device,
                                              res_2D);
                    host_bias_swish_f16(stream, res_2D.size(0), res_2D.size(1), res_2D.stride(0),
                                        res_2D.data_ptr(), b_device.data_ptr());

                    // Output is [N, T_out, C_out], contiguous
                    return res;
                } else {
                    auto res = torch::empty({chunk_size_out + 1, batch_size, 2, out_size},
                                            x.options());
                    res.index({0, Slice(), 1, Slice()}) = 0;
                    res.index({chunk_size_out, Slice(), 0, Slice()}) = 0;
                    auto res_TNC = res.slice(0, 1, chunk_size_out + 1).select(2, 1);
                    auto res_2D = res_TNC.view({-1, out_size});

                    auto tncw_mat = torch::empty({chunk_size_out, batch_size, in_size, window_size},
                                                 x.options());
                    host_window_ntcw_f16(stream, x.stride(0), x.stride(2), x.stride(1), batch_size,
                                         chunk_size_in, in_size, window_size, stride,
                                         tncw_mat.stride(1), tncw_mat.stride(0), tncw_mat.stride(2),
                                         tncw_mat.stride(3), x.data_ptr(), tncw_mat.data_ptr());
                    dorado::utils::matmul_f16(tncw_mat.view({-1, in_size * window_size}), w_device,
                                              res_2D);
                    host_bias_swish_f16(stream, res_2D.size(0), res_2D.size(1), res_2D.stride(0),
                                        res_2D.data_ptr(), b_device.data_ptr());

                    // Output is [T_out + 1, N, 2, C_out], contiguous, which serves as
                    // working memory for CuBLAS LSTM
                    return res;
                }
            } else
#endif
            {
                // Output is [N, T_out, C_out], non-contiguous
                return activation(conv(x)).transpose(1, 2);
            }
        }
        // Output is [N, C_out, T_out], contiguous
        return activation(conv(x));
    }

    Conv1d conv{nullptr};
    SiLU activation{nullptr};
    int in_size;
    int out_size;
    int window_size;
    int stride;
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

#if USE_CUDA_LSTM

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

struct CudaLSTMStackImpl : Module {
    CudaLSTMStackImpl(int layer_size_, int batch_size, int chunk_size) : layer_size(layer_size_) {
        rnn1 = register_module("rnn_1", CudaLSTM(layer_size, true));
        rnn2 = register_module("rnn_2", CudaLSTM(layer_size, false));
        rnn3 = register_module("rnn_3", CudaLSTM(layer_size, true));
        rnn4 = register_module("rnn_4", CudaLSTM(layer_size, false));
        rnn5 = register_module("rnn_5", CudaLSTM(layer_size, true));

        m_quantize = cuda_lstm_is_quantized(layer_size);

        if (m_quantize) {
            // chunk_size * batch_size can not be > 2**31 (2147483648).
            // For practical purposes this is currently always the case.
            _chunks = torch::empty({batch_size, 4}).to(torch::kInt32);
            _chunks.index({torch::indexing::Slice(), 0}) =
                    torch::arange(0, chunk_size * batch_size, chunk_size);
            _chunks.index({torch::indexing::Slice(), 2}) =
                    torch::arange(0, chunk_size * batch_size, chunk_size);
            _chunks.index({torch::indexing::Slice(), 1}) = chunk_size;
            _chunks.index({torch::indexing::Slice(), 3}) = 0;
        }

        if (layer_size == 96) {
            _host_run_lstm_fwd_quantized = host_run_lstm_fwd_quantized96;
            _host_run_lstm_rev_quantized = host_run_lstm_reverse_quantized96;
        } else if (layer_size == 128) {
            _host_run_lstm_fwd_quantized = host_run_lstm_fwd_quantized128;
            _host_run_lstm_rev_quantized = host_run_lstm_reverse_quantized128;
        }
    }

    bool _weights_rearranged = false;
    bool m_quantize;
    torch::Tensor _chunks;
    std::vector<torch::Tensor> _r_wih;
    std::vector<torch::Tensor> _quantized_buffers;
    std::vector<torch::Tensor> _quantization_scale_factors;
    quantized_lstm _host_run_lstm_fwd_quantized{nullptr};
    quantized_lstm _host_run_lstm_rev_quantized{nullptr};

    torch::Tensor forward_cublas(torch::Tensor in) {
        // input in is ([N, T, C], contiguity optional) or ([T+1, N, 2, C], contiguous) (see below)
        c10::cuda::CUDAGuard device_guard(in.device());
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        int chunk_size, batch_size;
        torch::Tensor mat_working_mem;
        bool input_is_working_mem = (in.dim() == 4 && in.size(2) == 2);
        if (input_is_working_mem) {
            mat_working_mem = in;
            chunk_size = in.size(0) - 1;
            batch_size = in.size(1);
            assert(layer_size == in.size(3));
            assert(in.is_contiguous());
        } else {
            batch_size = in.size(0);
            chunk_size = in.size(1);
            assert(layer_size == in.size(2));
            mat_working_mem =
                    torch::zeros({chunk_size + 1, batch_size, 2, layer_size}, in.options());
        }

        int gate_size = layer_size * 4;
        auto gate_buf = torch::empty({batch_size, gate_size}, in.options());

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

        auto working_mem_all = mat_working_mem.view({chunk_size + 1, batch_size, -1});
        auto working_mem_left = mat_working_mem.slice(0, 0, chunk_size).select(2, 0);
        auto working_mem_right = mat_working_mem.slice(0, 1, chunk_size + 1).select(2, 1);

        if (!input_is_working_mem) {
            // NOTE: `host_transpose_f16' does exactly what the commented out assignment
            // below would do, only ~5x faster (on A100)
            // working_mem_right = in.transpose(1, 0);
            host_transpose_f16(stream, in.data_ptr(), in.size(1), in.size(0), in.size(2),
                               in.stride(1), in.stride(0), in.stride(2),
                               working_mem_right.stride(0), working_mem_right.stride(1),
                               working_mem_right.stride(2), working_mem_right.data_ptr());
        }

        for (auto &rnn : {rnn1, rnn2, rnn3, rnn4, rnn5}) {
            auto state_buf = torch::zeros({batch_size, layer_size}, in.options());
            auto weights_cpu = rnn->weights.t().contiguous();
            auto weights = weights_cpu.to(in.device());
            auto bias = rnn->bias.to(in.device());
            for (int ts = 0; ts < chunk_size; ++ts) {
                auto timestep_in = working_mem_all[rnn->reverse ? (chunk_size - ts) : ts];
                auto timestep_out = rnn->reverse ? working_mem_left[chunk_size - ts - 1]
                                                 : working_mem_right[ts];

                // Timestep matrix mulitplication
                dorado::utils::matmul_f16(timestep_in, weights, gate_buf);
                host_lstm_step_f16(stream, batch_size, layer_size, bias.data_ptr(),
                                   gate_buf.data_ptr(), state_buf.data_ptr(),
                                   timestep_out.data_ptr());
            }
        }

        // Output is [N, T, C], non-contiguous
        return working_mem_left.transpose(1, 0);
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

    void rearrange_weights() {
        for (auto &rnn : {rnn1, rnn2, rnn3, rnn4, rnn5}) {
            rearrange_individual_weights(rnn->named_parameters()["weight_hh"]);
            rearrange_individual_weights(rnn->named_parameters()["weight_ih"]);
            _r_wih.push_back(rnn->named_parameters()["weight_ih"].transpose(0, 1).contiguous());
            rearrange_individual_weights(rnn->named_parameters()["bias_hh"]);
            rearrange_individual_weights(rnn->named_parameters()["bias_ih"]);
        }
        _weights_rearranged = true;
    }

    std::pair<torch::Tensor, torch::Tensor> quantize_tensor(torch::Tensor tensor,
                                                            int levels = 256) {
        //Qauntize a tensor to int8, returning per-channel scales and the quantized tensor
        //if weights have not been quantized we get some scaling
        tensor = tensor.transpose(0, 1).contiguous();
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

    void quantize_weights() {
        for (auto &rnn : {rnn1, rnn2, rnn3, rnn4, rnn5}) {
            auto [factors, quantized] = quantize_tensor(rnn->named_parameters()["weight_hh"]);
            _quantization_scale_factors.push_back(factors);
            _quantized_buffers.push_back(quantized);
        }
    }

    torch::Tensor forward_quantized(torch::Tensor x) {
        // Input x is [N, T, C], contiguity optional
        c10::cuda::CUDAGuard device_guard(x.device());

        x = x.contiguous();

        //If this is the fist time the forward method is being applied, do some startup
        if (m_quantize && !_weights_rearranged) {
            rearrange_weights();
            quantize_weights();
            _chunks = _chunks.to(x.device());
        }
        auto buffer = torch::matmul(x, _r_wih[0]);

        _host_run_lstm_rev_quantized(
                _chunks.data_ptr(), buffer.data_ptr(), _quantized_buffers[0].data_ptr(),
                rnn1->named_parameters()["bias_ih"].data_ptr(),
                _quantization_scale_factors[0].data_ptr(), x.data_ptr(), _chunks.size(0));

        buffer = torch::matmul(x, _r_wih[1]);

        _host_run_lstm_fwd_quantized(
                _chunks.data_ptr(), buffer.data_ptr(), _quantized_buffers[1].data_ptr(),
                rnn2->named_parameters()["bias_ih"].data_ptr(),
                _quantization_scale_factors[1].data_ptr(), x.data_ptr(), _chunks.size(0));

        buffer = torch::matmul(x, _r_wih[2]);

        _host_run_lstm_rev_quantized(
                _chunks.data_ptr(), buffer.data_ptr(), _quantized_buffers[2].data_ptr(),
                rnn3->named_parameters()["bias_ih"].data_ptr(),
                _quantization_scale_factors[2].data_ptr(), x.data_ptr(), _chunks.size(0));

        buffer = torch::matmul(x, _r_wih[3]);

        _host_run_lstm_fwd_quantized(
                _chunks.data_ptr(), buffer.data_ptr(), _quantized_buffers[3].data_ptr(),
                rnn4->named_parameters()["bias_ih"].data_ptr(),
                _quantization_scale_factors[3].data_ptr(), x.data_ptr(), _chunks.size(0));

        buffer = torch::matmul(x, _r_wih[4]);

        _host_run_lstm_rev_quantized(
                _chunks.data_ptr(), buffer.data_ptr(), _quantized_buffers[4].data_ptr(),
                rnn5->named_parameters()["bias_ih"].data_ptr(),
                _quantization_scale_factors[4].data_ptr(), x.data_ptr(), _chunks.size(0));

        // Output is [N, T, C], contiguous
        return x;
    }

    // Dispatch to different forward method depending on whether we use quantized LSTMs or not
    torch::Tensor forward(torch::Tensor x) {
        // Input x is [N, T, C], contiguity optional
        if (m_quantize) {
            // Output is [N, T, C], contiguous
            return forward_quantized(x);
        } else {
            // Output is [N, T, C], non-contiguous
            return forward_cublas(x);
        }
    }

    int layer_size;
    CudaLSTM rnn1{nullptr}, rnn2{nullptr}, rnn3{nullptr}, rnn4{nullptr}, rnn5{nullptr};
};

TORCH_MODULE(CudaLSTMStack);

#endif  // if USE_CUDA_LSTM

struct LSTMStackImpl : Module {
    LSTMStackImpl(int size, int batchsize, int chunksize) {
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

    LSTM rnn1{nullptr}, rnn2{nullptr}, rnn3{nullptr}, rnn4{nullptr}, rnn5{nullptr};
};

struct ClampImpl : Module {
    ClampImpl(float _min, float _max, bool _active) : min(_min), max(_max), active(_active){};

    torch::Tensor forward(torch::Tensor x) {
        if (active) {
            return x.clamp(min, max);
        } else {
            return x;
        }
    }

    bool active;
    float min, max;
};

TORCH_MODULE(LSTMStack);
TORCH_MODULE(LinearCRF);
TORCH_MODULE(Convolution);
TORCH_MODULE(Clamp);

template <class LSTMStackType>
struct CRFModelImpl : Module {
    CRFModelImpl(const CRFModelConfig &config, bool expand_blanks, int batch_size, int chunk_size) {
        conv1 = register_module("conv1", Convolution(config.num_features, config.conv, 5, 1));
        clamp1 = Clamp(-0.5, 3.5, config.clamp);
        conv2 = register_module("conv2", Convolution(config.conv, 16, 5, 1));
        clamp2 = Clamp(-0.5, 3.5, config.clamp);
        conv3 = register_module("conv3", Convolution(16, config.insize, 19, config.stride, true));
        clamp3 = Clamp(-0.5, 3.5, config.clamp);

        rnns = register_module(
                "rnns", LSTMStackType(config.insize, batch_size, chunk_size / config.stride));

        if (config.out_features.has_value()) {
            // The linear layer is decomposed into 2 matmuls.
            const int decomposition = config.out_features.value();
            linear1 = register_module("linear1", Linear(config.insize, decomposition));
            linear2 = register_module(
                    "linear2", Linear(LinearOptions(decomposition, config.outsize).bias(false)));
            clamp4 = Clamp(-5.0, 5.0, config.clamp);
            encoder = Sequential(conv1, clamp1, conv2, clamp2, conv3, clamp3, rnns, linear1,
                                 linear2, clamp4);
        } else if ((config.conv == 16) && (config.num_features == 1)) {
            linear1 = register_module(
                    "linear1", Linear(LinearOptions(config.insize, config.outsize).bias(false)));
            clamp4 = Clamp(-5.0, 5.0, config.clamp);
            encoder =
                    Sequential(conv1, clamp1, conv2, clamp2, conv3, clamp3, rnns, linear1, clamp4);
        } else {
            linear = register_module("linear1", LinearCRF(config.insize, config.outsize));
            encoder = Sequential(conv1, conv2, conv3, rnns, linear);
        }
    }

    void load_state_dict(const std::vector<torch::Tensor> &weights) {
        utils::load_state_dict(*this, weights);
    }

    torch::Tensor forward(torch::Tensor x) {
        nvtx3::scoped_range loop{"nn_forward"};
        if (x.device() == torch::kCPU) {
            // Output is [T, N, C], which CPU decoding requires.
            return encoder->forward(x).transpose(0, 1);
        }
        // Output is [N, T, C]
        return encoder->forward(x);
    }

    LSTMStackType rnns{nullptr};
    LinearCRF linear{nullptr};
    Linear linear1{nullptr}, linear2{nullptr};
    Sequential encoder{nullptr};
    Convolution conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    Clamp clamp1{nullptr}, clamp2{nullptr}, clamp3{nullptr}, clamp4{nullptr};
};

#if USE_CUDA_LSTM
using CudaCRFModelImpl = CRFModelImpl<CudaLSTMStack>;
TORCH_MODULE(CudaCRFModel);
#endif

using CpuCRFModelImpl = CRFModelImpl<LSTMStack>;
TORCH_MODULE(CpuCRFModel);

}  // namespace nn

CRFModelConfig load_crf_model_config(const std::filesystem::path &path) {
    const auto config_toml = toml::parse(path / "config.toml");

    CRFModelConfig config;
    config.qscale = 1.0f;
    config.qbias = 0.0f;

    if (config_toml.contains("qscore")) {
        const auto &qscore = toml::find(config_toml, "qscore");
        config.qbias = toml::find<float>(qscore, "bias");
        config.qscale = toml::find<float>(qscore, "scale");
    } else {
        spdlog::debug("> no qscore calibration found");
    }

    config.conv = 4;
    config.insize = 0;
    config.stride = 1;
    config.bias = true;
    config.clamp = false;

    // The encoder scale only appears in pre-v4 models.  In v4 models
    // the value of 1 is used.
    config.scale = 1.0f;

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

    return config;
}

std::vector<torch::Tensor> load_crf_model_weights(const std::filesystem::path &dir,
                                                  bool decomposition,
                                                  bool bias) {
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

    if (bias) {
        tensors.push_back("9.linear.bias.tensor");
    }

    if (decomposition) {
        tensors.push_back("10.linear.weight.tensor");
    }

    return utils::load_tensors(dir, tensors);
}

ModuleHolder<AnyModule> load_crf_model(const std::filesystem::path &path,
                                       const CRFModelConfig &model_config,
                                       const int batch_size,
                                       const int chunk_size,
                                       const torch::TensorOptions &options) {
#if USE_CUDA_LSTM
    if (options.device() != torch::kCPU) {
        const bool expand_blanks = false;
        auto model = nn::CudaCRFModel(model_config, expand_blanks, batch_size, chunk_size);
        return populate_model(model, path, options, model_config.out_features.has_value(),
                              model_config.bias);
    } else
#endif
    {
        const bool expand_blanks = true;
        auto model = nn::CpuCRFModel(model_config, expand_blanks, batch_size, chunk_size);
        return populate_model(model, path, options, model_config.out_features.has_value(),
                              model_config.bias);
    }
}

}  // namespace dorado
