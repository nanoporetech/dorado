#include "CRFModel.h"

#include "../utils/module_utils.h"
#include "../utils/tensor_utils.h"

#ifndef __APPLE__
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

extern "C" {
#include "koi.h"
}
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define USE_CUDA_LSTM 1
#else
#define USE_CUDA_LSTM 0
#endif

#include <math.h>
#include <toml.hpp>
#include <torch/torch.h>

#include <string>

using namespace torch::nn;
namespace F = torch::nn::functional;
using Slice = torch::indexing::Slice;
using quantized_lstm = std::function<int(void *, void *, void *, void *, void *, void *, int)>;

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
        auto T = x.size(0);
        auto N = x.size(1);
#if USE_CUDA_LSTM
        // Optimised version of the #else branch for CUDA devices
        c10::cuda::CUDAGuard device_guard(x.device());
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        // make sure input is NTC in memory
        x = x.transpose(0, 1).contiguous().reshape({T * N, -1});
        auto scores = torch::matmul(x, linear->weight.t());
        host_bias_tanh_scale_f16(stream, N * T, scores.size(1), scale, scores.data_ptr(),
                                 linear->bias.data_ptr());
        scores = scores.view({N, T, -1}).transpose(0, 1);  // logical order TNC, memory order NTC

#else  // if USE_CUDA_LSTM
        auto scores = activation(linear(x)) * scale;
#endif

        if (expand_blanks == true) {
            scores = scores.contiguous();
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

torch::TensorOptions get_tensor_options(CudaLSTM lstm) {
    return lstm->named_parameters()["weight_hh"].options();
};

struct LSTMStackImpl : Module {
    LSTMStackImpl(int layer_size_, int batch_size, int chunk_size) : layer_size(layer_size_) {
        rnn1 = register_module("rnn_1", CudaLSTM(layer_size, true));
        _rnns.push_back(rnn1);
        rnn2 = register_module("rnn_2", CudaLSTM(layer_size, false));
        _rnns.push_back(rnn2);
        rnn3 = register_module("rnn_3", CudaLSTM(layer_size, true));
        _rnns.push_back(rnn3);
        rnn4 = register_module("rnn_4", CudaLSTM(layer_size, false));
        _rnns.push_back(rnn4);
        rnn5 = register_module("rnn_5", CudaLSTM(layer_size, true));
        _rnns.push_back(rnn5);
        _tensor_options = get_tensor_options(rnn1);

        m_batch_size = batch_size;
        m_quantize = ((layer_size == 96) || (layer_size == 128));

        if (m_quantize) {
            // TODO: fix this hardcoded 5 with stride
            int block_size = chunk_size / 5;

            // Create some working buffers which are needed for quantized kernels
            _buffer1 = torch::empty({batch_size, block_size, layer_size}).to(_tensor_options);
            _buffer2 = torch::empty({batch_size, block_size, layer_size}).to(_tensor_options);

            _chunks = torch::empty({batch_size, 4}).to(_tensor_options).to(torch::kI32);
            _chunks.index({torch::indexing::Slice(), 0}) =
                    torch::arange(0, block_size * batch_size, block_size);
            _chunks.index({torch::indexing::Slice(), 2}) =
                    torch::arange(0, block_size * batch_size, block_size);
            _chunks.index({torch::indexing::Slice(), 1}) = block_size;
            _chunks.index({torch::indexing::Slice(), 2}) = 0;
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
    int m_batch_size;
    std::vector<CudaLSTM> _rnns;
    std::vector<torch::Tensor> _r_wih;
    std::vector<torch::Tensor> _quantized_buffers;
    std::vector<torch::Tensor> _quantization_scale_factors;
    torch::Tensor _buffer1;
    torch::Tensor _buffer2;
    torch::Tensor _chunks;
    torch::TensorOptions _tensor_options;
    quantized_lstm _host_run_lstm_fwd_quantized, _host_run_lstm_rev_quantized;

    torch::Tensor forward_cublas(torch::Tensor in) {
        c10::cuda::CUDAGuard device_guard(in.device());
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        int chunk_size = in.size(0);
        int batch_size = in.size(1);
        assert(layer_size == in.size(2));
        int gate_size = layer_size * 4;

        // We need some extra working memory to run the LSTM layers. By making it `thread_local`
        // this will work with multiple runners (i.e. multiple threads).
        thread_local torch::Tensor mat_working_mem, gate_buf;
        thread_local int working_mem_chunk_size = 0;
        thread_local int max_batch_size = 0;
        if (working_mem_chunk_size != chunk_size || max_batch_size < batch_size) {
            if (max_batch_size < batch_size) {
                gate_buf = torch::empty({batch_size, gate_size}, in.options()).contiguous();
                max_batch_size = batch_size;
            }
            working_mem_chunk_size = chunk_size;
            mat_working_mem =
                    torch::zeros({chunk_size + 1, batch_size, 2, layer_size}, in.options())
                            .contiguous();
        }
        mat_working_mem.to(in.device());
        gate_buf.to(in.device());

        // Working memory is laid out as [T+1][N][2][C] in memory, where the 2 serves to
        // interleave input and output for each LSTM layer in a specific way. The reverse LSTM
        // layers (rnn1, rnn3, rnn5) use right as input and left as output, whereas the forward
        // LSTM layers (rnn2, rnn4) use left as input and right as output.
        //
        // The interleaving means that x(t) and h(t-1), i.e. the input for the current timestep
        // and the output of the previous timestep, appear concatenated in memory and we can
        // perform a single matmul with the concatenated WU matrix
        // Note that both working_mem[chunk_size][:][0][:] and working_mem[0][:][1][0] remain
        // all zeroes, representing the initial LSTM state h(-1) in either direction.

        auto working_mem_all = mat_working_mem.view({chunk_size + 1, batch_size, -1});
        auto working_mem_left = mat_working_mem.slice(0, 0, chunk_size).select(2, 0);
        auto working_mem_right = mat_working_mem.slice(0, 1, chunk_size + 1).select(2, 1);

        // NOTE: `host_transpose_f16' does exactly what the commented out assignment
        // below would do, only ~5x faster (on A100)
        // working_mem_right = in;
        host_transpose_f16(stream, in.data_ptr(), in.size(0), in.size(1), in.size(2), in.stride(0),
                           in.stride(1), in.stride(2), working_mem_right.stride(0),
                           working_mem_right.stride(1), 1, working_mem_right.data_ptr());

        for (auto &rnn : {rnn1, rnn2, rnn3, rnn4, rnn5}) {
            auto state_buf = torch::zeros({batch_size, layer_size}, in.options()).contiguous();
            auto weights_cpu = rnn->weights.t().contiguous();
            auto weights = weights_cpu.to(in.device());
            auto bias = rnn->bias.to(in.device());
            for (int ts = 0; ts < chunk_size; ++ts) {
                auto timestep_in = working_mem_all[rnn->reverse ? (chunk_size - ts) : ts];
                auto timestep_out = rnn->reverse ? working_mem_left[chunk_size - ts - 1]
                                                 : working_mem_right[ts];

                // Timestep matrix mulitplication (using cublasGemmEx, as using torch::matmul
                // as below is a bit slower on A100 for some reason)
                // gate_buf = torch::matmul(timestep_in, weights);
                constexpr uint16_t HALF_ZERO = 0;      // 0.0 in __half format
                constexpr uint16_t HALF_ONE = 0x3C00;  // 1.0 in __half format
                auto res = cublasGemmEx(
                        at::cuda::getCurrentCUDABlasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, gate_size,
                        batch_size, 2 * layer_size, &HALF_ONE, (const void *)weights.data_ptr(),
                        CUDA_R_16F, gate_size, (const void *)timestep_in.data_ptr(), CUDA_R_16F,
                        2 * layer_size, &HALF_ZERO, (void *)gate_buf.data_ptr(), CUDA_R_16F,
                        gate_size, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (res != CUBLAS_STATUS_SUCCESS) {
                    std::cerr << "CuBLAS error " << int(res) << std::endl;
                    exit(EXIT_FAILURE);
                }
                host_lstm_step_f16(stream, batch_size, layer_size, bias.data_ptr(),
                                   gate_buf.data_ptr(), state_buf.data_ptr(),
                                   timestep_out.data_ptr());
            }
        }

        // NOTE: we return a view to a thread_local tensor here, meaning we could get weird
        // results if we called this method again on the same thread before consuming the
        // tensor contents. For that to happen we'd have to have two consecutive LSTMStacks,
        // which doesn't make much sense. So this should be safe in practice.
        return mat_working_mem.index({Slice(0, chunk_size), Slice(), 0});
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
        for (auto rnn : _rnns) {
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
        //if weights have not been quantized we get some scalin
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
        for (auto rnn : _rnns) {
            std::pair<torch::Tensor, torch::Tensor> quantization_results =
                    quantize_tensor(rnn->named_parameters()["weight_hh"]);
            _quantization_scale_factors.push_back(quantization_results.first);
            _quantized_buffers.push_back(quantization_results.second);
        }
    }

    torch::Tensor forward_quantized(torch::Tensor x) {
        //If this is the fist time the forward method is being applied, do some startup
        if (m_quantize && !_weights_rearranged) {
            rearrange_weights();
            quantize_weights();
            // TODO: For multi-GPU this will need to be smarter
            _buffer1 = _buffer1.to(x.device());
            _buffer2 = _buffer2.to(x.device());
            _chunks = _chunks.to(x.device());
        }

        x = x.permute({1, 0, 2}).contiguous();  // data needs to be in NTC format.

        _buffer1 = torch::matmul(x, _r_wih[0]);

        _host_run_lstm_rev_quantized(
                _chunks.data_ptr(), _buffer1.data_ptr(), _quantized_buffers[0].data_ptr(),
                _rnns[0]->named_parameters()["bias_ih"].data_ptr(),
                _quantization_scale_factors[0].data_ptr(), _buffer2.data_ptr(), m_batch_size);

        _buffer1 = torch::matmul(_buffer2, _r_wih[1]);

        _host_run_lstm_fwd_quantized(
                _chunks.data_ptr(), _buffer1.data_ptr(), _quantized_buffers[1].data_ptr(),
                _rnns[1]->named_parameters()["bias_ih"].data_ptr(),
                _quantization_scale_factors[1].data_ptr(), _buffer2.data_ptr(), m_batch_size);

        _buffer1 = torch::matmul(_buffer2, _r_wih[2]);

        _host_run_lstm_rev_quantized(
                _chunks.data_ptr(), _buffer1.data_ptr(), _quantized_buffers[2].data_ptr(),
                _rnns[2]->named_parameters()["bias_ih"].data_ptr(),
                _quantization_scale_factors[2].data_ptr(), _buffer2.data_ptr(), m_batch_size);

        _buffer1 = torch::matmul(_buffer2, _r_wih[3]);

        _host_run_lstm_fwd_quantized(
                _chunks.data_ptr(), _buffer1.data_ptr(), _quantized_buffers[3].data_ptr(),
                _rnns[3]->named_parameters()["bias_ih"].data_ptr(),
                _quantization_scale_factors[3].data_ptr(), _buffer2.data_ptr(), m_batch_size);

        _buffer1 = torch::matmul(_buffer2, _r_wih[4]);

        _host_run_lstm_rev_quantized(
                _chunks.data_ptr(), _buffer1.data_ptr(), _quantized_buffers[4].data_ptr(),
                _rnns[4]->named_parameters()["bias_ih"].data_ptr(),
                _quantization_scale_factors[4].data_ptr(), _buffer2.data_ptr(), m_batch_size);

        return _buffer2.permute({1, 0, 2}).contiguous();
    }

    // Dispatch to different forward method depending on whether we use quantized LSTMs or not
    torch::Tensor forward(torch::Tensor x) {
        if (m_quantize) {
            return forward_quantized(x);
        } else {
            return forward_cublas(x);
        }
    }

    int layer_size;
    CudaLSTM rnn1{nullptr}, rnn2{nullptr}, rnn3{nullptr}, rnn4{nullptr}, rnn5{nullptr};
};

#else  // if USE_CUDA_LSTM

struct LSTMStackImpl : Module {
    LSTMStackImpl(int size, int batchsize, int chunksize) {
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
    CRFModelImpl(int size,
                 int outsize,
                 int stride,
                 bool expand_blanks,
                 int batch_size,
                 int chunk_size) {
        conv1 = register_module("conv1", Convolution(1, 4, 5, 1));
        conv2 = register_module("conv2", Convolution(4, 16, 5, 1));
        conv3 = register_module("conv3", Convolution(16, size, 19, stride));
        permute = register_module("permute", Permute());
        rnns = register_module("rnns", LSTMStack(size, batch_size, chunk_size));
        linear = register_module("linear", LinearCRF(size, outsize));
        linear->expand_blanks = expand_blanks;
        encoder = Sequential(conv1, conv2, conv3, permute, rnns, linear);
    }

    void load_state_dict(const std::vector<torch::Tensor> &weights) {
        ::utils::load_state_dict(*this, weights);
    }

    torch::Tensor forward(torch::Tensor x) { return encoder->forward(x); }

    std::vector<torch::Tensor> load_weights(const std::filesystem::path &dir) {
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

                "9.linear.weight.tensor",    "9.linear.bias.tensor"};

        return ::utils::load_tensors(dir, tensors);
    }

    Permute permute{nullptr};
    LSTMStack rnns{nullptr};
    LinearCRF linear{nullptr};
    Sequential encoder{nullptr};
    Convolution conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
};

TORCH_MODULE(CRFModel);

std::tuple<ModuleHolder<AnyModule>, size_t> load_crf_model(const std::filesystem::path &path,
                                                           int batch_size,
                                                           int chunk_size,
                                                           torch::TensorOptions options) {
    auto config = toml::parse(path / "config.toml");

    const auto &encoder = toml::find(config, "encoder");
    const auto stride = toml::find<int>(encoder, "stride");
    const auto insize = toml::find<int>(encoder, "features");

    const auto &global_norm = toml::find(config, "global_norm");
    const auto state_len = toml::find<int>(global_norm, "state_len");
    int outsize = pow(4, state_len) * 4;
    bool expand = options.device_opt().value() == torch::kCPU;

    auto model = CRFModel(insize, outsize, stride, expand, batch_size, chunk_size);
    auto state_dict = model->load_weights(path);
    model->load_state_dict(state_dict);
    model->to(options.dtype_opt().value().toScalarType());
    model->to(options.device_opt().value());
    model->eval();

    auto module = AnyModule(model);
    auto holder = ModuleHolder<AnyModule>(module);

    return {holder, static_cast<size_t>(stride)};
}
