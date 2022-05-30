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
