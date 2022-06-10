#include "CRFModel.h"

#include "../utils/tensor_utils.h"

#include <math.h>
#include <toml.hpp>
#include <torch/torch.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

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

TORCH_MODULE(Permute);
TORCH_MODULE(LSTMStack);
TORCH_MODULE(LinearCRF);
TORCH_MODULE(Convolution);

#if USE_CUSPARSE

#define CUSPARSE_CHECK(X) { cusparseStatus_t res = X; if (res != CUSPARSE_STATUS_SUCCESS) { \
  printf("CuSPARSELt returned error code %d, line(%d)\n", int(res), __LINE__); \
  exit(EXIT_FAILURE); \
}}

struct CusparseLSTMImpl : Module {
    CusparseLSTMImpl(int layer_size, bool reverse_) : reverse(reverse_) {
        // TODO: do we need to specify .device("gpu")?
        auto options = torch::TensorOptions().dtype(torch::kFloat16);
        weight = torch::empty({layer_size * 4, layer_size * 2}, options).contiguous();
        auto weight_ih = weight.index({Slice(), Slice(0, layer_size));
        auto weight_hh = weight.index({Slice(), Slice(layer_size, 2 * layer_size));
        if (reverse) {
            std::swap(weight_ih, weight_hh);
        }
        bias = torch::empty({layer_size * 4}, options).contiguous();

        register_parameter("weight_ih", weight_ih, false);
        register_parameter("weight_hh", weight_hh, false);
        register_parameter("bias_ih", bias, false);
        // TODO: do we need to register "bias_hh"?
    }

    torch::Tensor weight, bias;
    bool reverse;
};

TORCH_MODULE(CusparseLSTM);

struct CusparseLSTMBlockImpl : Module {
    CusparseLSTMBlockImpl(int chunk_size_,
                   int batch_size_,
                   int stride_,
                   int layer_size_,
                   int out_size_)
            : in_chunk_size(chunk_size_),
              stride(stride_),
              batch_size(batch_size_),
              layer_size(layer_size_),
              out_size(out_size_) {
        lstm_chunk_size = in_chunk_size / stride;

//        reorder_input_cps = make_cps(device, "reorder_input");
//        reorder_output_cps = make_cps(device, "reorder_output");
//        reorder_weights_cps = make_cps(device, "reorder_weights");

        // TODO: we can reuse some of these matrices
        mat_working_mem = create_buffer(
                device, size_t(lstm_chunk_size + 2) * batch_size * layer_size * sizeof(ftype));
        mat_state = create_buffer(device, batch_size * layer_size * sizeof(ftype));
        mat_temp_result = create_buffer(device, batch_size * layer_size * 4 * sizeof(ftype));

        rnn1 = register_module("rnn_1", CusparseLSTM(layer_size, true, device));
        rnn2 = register_module("rnn_2", CusparseLSTM(layer_size, false, device));
        rnn3 = register_module("rnn_3", CusparseLSTM(layer_size, true, device));
        rnn4 = register_module("rnn_4", CusparseLSTM(layer_size, false, device));
        rnn5 = register_module("rnn_5", CusparseLSTM(layer_size, true, device));
    }

    void load_weights() {
        for (auto &rnn : {rnn1, rnn2, rnn3, rnn4, rnn5}) {
            auto params = rnn->named_parameters();

            auto t_w = *params.find("weight_ih");
            auto t_u = *params.find("weight_hh");
            auto t_b = *params.find("bias_ih");

            // reorder from IFGO to GIFO (2, 0, 1, 3)
            t_w = t_w.reshape({4, layer_size, layer_size}).transpose(1, 2);
            t_w = torch::concat({t_w[2], t_w[0], t_w[1], t_w[3]}, 1);

            t_u = t_u.reshape({4, layer_size, layer_size}).transpose(1, 2);
            t_u = torch::concat({t_u[2], t_u[0], t_u[1], t_u[3]}, 1);

            t_b = t_b.reshape({4, layer_size});
            t_b = torch::concat({t_b[2], t_b[0], t_b[1], t_b[3]});

            t_w = t_w.flatten(0, -1).contiguous();
            t_u = t_u.flatten(0, -1).contiguous();
            t_b = t_b.flatten(0, -1).contiguous();

            std::vector<MTL::Buffer *> buffers{args[rnn->reverse], mtl_for_tensor(t_w),
                                               mtl_for_tensor(t_u), mtl_for_tensor(t_b),
                                               rnn->mat_weights};
            launch_kernel(reorder_weights_cps, command_queue, buffers, kernel_thread_groups, 256);
        }
    }

    torch::Tensor forward(torch::Tensor in) {
        auto options = torch::TensorOptions().dtype(torch::kFloat16);
        auto mat_working_mem = torch::zeros({in.size(0) + 1, in.size(1), in.size(2) * 2}, options).contiguous();
        if (rnn1.reverse) {
            mat_working_mem.slice({Slice(0, in.size(0)), Slice(), Slice(0, in.size(2))}) = in;
        } else {
            mat_working_mem.slice({Slice(1, in.size(0)), Slice(), Slice(in.size(2), 2 * in.size(2))}) = in;
        }

        torch::Tensor out = torch::empty({lstm_chunk_size, batch_size, out_size});
        for (auto &rnn : {rnn1, rnn2, rnn3, rnn4, rnn5}) {
            std::vector<MTL::Buffer *> buffers{args[rnn->reverse], mat_working_mem,
                                               rnn->mat_weights, mat_state, mat_temp_result};
            launch_kernel_no_wait(lstm_cps[rnn->reverse], command_buffer, buffers,
                                  kernel_thread_groups, kernel_simd_groups * 32);
        }
        return out;
    }

    int in_chunk_size, lstm_chunk_size, stride, batch_size, layer_size, out_size,
            kernel_thread_groups, kernel_simd_groups;
    CusparseLSTM rnn1{nullptr}, rnn2{nullptr}, rnn3{nullptr}, rnn4{nullptr}, rnn5{nullptr};
};

TORCH_MODULE(CusparseLSTMBlock);

#endif // USE_CUSPARSE

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

ModuleHolder<AnyModule> load_crf_model(const std::string& path,
                                       int batch_size,
                                       int chunk_size,
                                       torch::TensorOptions options) {
    auto config = toml::parse(path + "/config.toml");

    const auto& encoder = toml::find(config, "encoder");
    const auto stride = toml::find<int>(encoder, "stride");
    const auto insize = toml::find<int>(encoder, "features");

    const auto& global_norm = toml::find(config, "global_norm");
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
