#include "ModBaseConvLSTMModel.h"

#include "ModBaseModelUtils.h"
#include "ModsConv.h"
#include "torch_utils/gpu_profiling.h"
#include "torch_utils/module_utils.h"

namespace dorado::modbase::model {

ModBaseConvLSTMModelImpl::ModBaseConvLSTMModelImpl(
        const config::modbase::ConvLSTMModelConfig& config)
        : m_is_conv_lstm_v2(false) {
    initialise(config.general);
}

ModBaseConvLSTMModelImpl::ModBaseConvLSTMModelImpl(
        const config::modbase::ConvLSTMV2ModelConfig& config)
        : m_is_conv_lstm_v2(true) {
    initialise(config.general);
}

void ModBaseConvLSTMModelImpl::initialise(const config::modbase::CommonModelParams& params) {
    const int signal_stride = params.stride;
    // conv_lstm_v2 models are padded to ensure the output shape is nicely indexable by the stride
    sig_conv1 = register_module("sig_conv1", ModsConv(1, 4, 5, 1, m_is_conv_lstm_v2 ? 2 : 0));
    sig_conv2 = register_module("sig_conv2", ModsConv(4, 16, 5, 1, m_is_conv_lstm_v2 ? 2 : 0));
    sig_conv3 = register_module(
            "sig_conv3", ModsConv(16, params.size, 9, signal_stride, m_is_conv_lstm_v2 ? 4 : 0));

    seq_conv1 = register_module("seq_conv1",
                                ModsConv(params.kmer_len * 4, 16, 5, 1, m_is_conv_lstm_v2 ? 2 : 0));
    seq_conv2 = register_module(
            "seq_conv2", ModsConv(16, params.size, 13, params.stride, m_is_conv_lstm_v2 ? 6 : 0));

    merge_conv1 = register_module(
            "merge_conv1", ModsConv(params.size * 2, params.size, 5, 1, m_is_conv_lstm_v2 ? 2 : 0));

    lstm1 = register_module("lstm1",
                            torch::nn::LSTM(torch::nn::LSTMOptions(params.size, params.size)));
    lstm2 = register_module("lstm2",
                            torch::nn::LSTM(torch::nn::LSTMOptions(params.size, params.size)));

    linear = register_module("linear", torch::nn::Linear(params.size, params.num_out));
    activation = register_module("activation", torch::nn::SiLU());
}

at::Tensor ModBaseConvLSTMModelImpl::forward(at::Tensor sigs, at::Tensor seqs) {
    // INPUT sigs: NCT & seqs: NTC
    {
        utils::ScopedProfileRange spr("sig convs", 2);
        sigs = sig_conv1(sigs);
        sigs = sig_conv2(sigs);
        sigs = sig_conv3(sigs);
    }

    // We are supplied one hot encoded sequences as (batch, signal, kmer_len * base_count) int8.
    // We need (batch, kmer_len * base_count, signal) and a dtype compatible with the float16
    // weights.
    const auto conv_dtype = (seqs.device() == torch::kCPU) ? torch::kFloat32 : torch::kFloat16;

    // seqs: NTC -> NCT
    seqs = seqs.permute({0, 2, 1}).to(conv_dtype);
    {
        utils::ScopedProfileRange spr("seq convs", 2);
        seqs = seq_conv1(seqs);
        seqs = seq_conv2(seqs);
    }

    // z: NCT
    auto z = torch::cat({sigs, seqs}, 1);
    {
        utils::ScopedProfileRange spr("merge convs", 2);
        // z: NCT -> TNC
        z = merge_conv1(z).permute({2, 0, 1});
    }
    {
        utils::ScopedProfileRange spr("lstm1", 2);
        auto [z1, h1] = lstm1(z);
        z = activation(z1).flip(0);
    }
    {
        utils::ScopedProfileRange spr("lstm2", 2);
        auto [z2, h2] = lstm2(z);
        z = activation(z2).flip(0);
    }

    if (m_is_conv_lstm_v2) {
        utils::ScopedProfileRange spr("chunk_linear", 2);
        // TNC -> NTC
        z = linear(z.permute({1, 0, 2})).softmax(2).flatten(1);
    } else {
        utils::ScopedProfileRange spr("context_linear", 2);
        // Take the final time step: TNC -> tNC -> NC
        z = z.index({-1}).permute({0, 1});
        z = linear(z).softmax(1);
    }
    return z;
}

void ModBaseConvLSTMModelImpl::load_state_dict(const std::vector<at::Tensor>& weights) {
    utils::load_state_dict(*this, weights);
}

std::vector<at::Tensor> ModBaseConvLSTMModelImpl::load_weights(const std::filesystem::path& dir) {
    return utils::load_tensors(dir, load_modbase_conv_lstm_weights(false));
}

}  // namespace dorado::modbase::model
