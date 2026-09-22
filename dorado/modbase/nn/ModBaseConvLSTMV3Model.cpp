#include "ModBaseConvLSTMV3Model.h"

#include "ModBaseModelUtils.h"
#include "torch_utils/gpu_profiling.h"
#include "torch_utils/module_utils.h"

#include <optional>
#include <stdexcept>
#include <utility>

#if DORADO_CUDA_BUILD
#include "nn/KoiUtils.h"
#include "torch_utils/tensor_utils.h"
#include "utils/log_utils.h"

#include <c10/cuda/CUDAGuard.h>
#endif

namespace dorado::modbase::model {

ModBaseConvLSTMV3ModelImpl::ModBaseConvLSTMV3ModelImpl(
        const config::modbase::ConvLSTMV3ModelConfig& config)
        : model_config(config) {
    const auto& sig_cvs = model_config.modules.signal_convs;
    const auto& seq_cvs = model_config.modules.sequence_convs;
    const auto& merge_cv = model_config.modules.merge_conv;
    const auto& lstms = model_config.modules.lstms;
    const auto& ll = model_config.modules.linear;
    const auto& lu = model_config.modules.upsample;

    if (sig_cvs.size() != 3) {
        throw std::runtime_error("ModBaseConvLSTMV3Model expected 3 signal convolutions");
    }
    if (seq_cvs.size() != 2) {
        throw std::runtime_error("ModBaseConvLSTMV3Model expected 2 sequence convolutions");
    }
    if (lstms.size() != 2) {
        throw std::runtime_error("ModBaseConvLSTMV3Model expected 2 lstms");
    }

    sig_conv1 = register_module("sig_conv1", ModsConv(sig_cvs.at(0)));
    sig_conv2 = register_module("sig_conv2", ModsConv(sig_cvs.at(1)));
    sig_conv3 = register_module("sig_conv3", ModsConv(sig_cvs.at(2)));

    seq_conv1 = register_module("seq_conv1", ModsConv(seq_cvs.at(0)));
    seq_conv2 = register_module("seq_conv2", ModsConv(seq_cvs.at(1)));

    merge_conv1 = register_module("merge_conv1", ModsConv(merge_cv));

    lstm1 = register_module(
            "lstm1", torch::nn::LSTM(torch::nn::LSTMOptions(lstms.at(0).size, lstms.at(0).size)));
    lstm2 = register_module(
            "lstm2", torch::nn::LSTM(torch::nn::LSTMOptions(lstms.at(1).size, lstms.at(1).size)));

    linear = register_module("linear", torch::nn::Linear(ll.in_size, ll.out_size));

    if (lu.has_value()) {
        upsample = register_module("upsample", nn::LinearUpsample(lu.value()));
    }
}

at::Tensor ModBaseConvLSTMV3ModelImpl::forward(at::Tensor sigs, at::Tensor seqs) {
    // INPUT sigs: NCT & seqs: NTC
    {
        utils::ScopedProfileRange spr("sig convs", 2);
        sigs = sig_conv1(sigs);
        sigs = sig_conv2(sigs);
        sigs = sig_conv3(sigs);
    }

    // We are supplied one hot encoded sequences as (batch, signal/stride, kmer_len * base_count)
    // int8. We need (batch, kmer_len * base_count, signal/stride) and a dtype compatible with the
    // float16 weights.
    const auto conv_dtype = (seqs.device() == torch::kCPU) ? torch::kFloat32 : torch::kFloat16;

    // seqs: NTC -> NCT
    seqs = seqs.permute({0, 2, 1}).to(conv_dtype);
    {
        utils::ScopedProfileRange spr("seq convs", 2);
        seqs = seq_conv1(seqs);
        seqs = seq_conv2(seqs);
    }

    // z: NCT -> N(2C)T
    auto z = torch::cat({sigs, seqs}, 1);
    {
        utils::ScopedProfileRange spr("merge convs", 2);
        z = merge_conv1(z).permute({2, 0, 1});  // NCT -> TNC
    }
    {
        utils::ScopedProfileRange spr("lstm1", 2);
        z = std::get<0>(lstm1(z)).flip(0);  // TNC -> T'NC
    }
    {
        utils::ScopedProfileRange spr("lstm2", 2);
        z = std::get<0>(lstm2(z));  // T'NC
    }
    {
        utils::ScopedProfileRange spr("linear", 2);
        z = linear(z).flip(0).permute({1, 0, 2});  // T'NC -> NTC
    }
    if (model_config.modules.upsample.has_value()) {
        utils::ScopedProfileRange spr2("upsample", 2);
        z = upsample->forward(z);  // NTC -> N(sf*T)C
    }
    utils::ScopedProfileRange spr2("softmax", 2);
    // NTC -> N(TC)
    return z.softmax(2).flatten(1);
}

void ModBaseConvLSTMV3ModelImpl::load_state_dict(const std::vector<at::Tensor>& weights) {
    utils::load_state_dict(*this, weights);
}

std::vector<at::Tensor> ModBaseConvLSTMV3ModelImpl::load_weights(const std::filesystem::path& dir) {
    return utils::load_tensors(
            dir, load_modbase_conv_lstm_weights(model_config.modules.upsample.has_value()));
}

#if DORADO_CUDA_BUILD

namespace {

std::pair<int64_t, int64_t> get_chunked_sequence_input_TC(
        const config::modbase::ConvLSTMV3ModelConfig& config) {
    return {config.chunked.chunk_size / (config.general.stride / config.sequence_stride),
            config.data.context.kmer_len * 4};
}

std::pair<int64_t, int64_t> get_chunked_signal_input_TC(
        const config::modbase::ConvLSTMV3ModelConfig& config) {
    return {config.chunked.chunk_size, 1};
}

}  // namespace

ModBaseConvLSTMV3CUDAModelImpl::ModBaseConvLSTMV3CUDAModelImpl(
        const config::modbase::ConvLSTMV3ModelConfig& config,
        const int batch_size_)
        : model_config(config), batch_size(batch_size_) {
    const auto& sig_cvs = model_config.modules.signal_convs;
    const auto& seq_cvs = model_config.modules.sequence_convs;
    const auto& merge_cv = model_config.modules.merge_conv;
    const auto& lstms = model_config.modules.lstms;
    const auto& ll = model_config.modules.linear;
    const auto& lu = model_config.modules.upsample;

    if (sig_cvs.size() != 3) {
        throw std::runtime_error("ModBaseConvLSTMV3Model expected 3 signal convolutions");
    }
    if (seq_cvs.size() != 2) {
        throw std::runtime_error("ModBaseConvLSTMV3Model expected 2 sequence convolutions");
    }
    if (lstms.size() != 2) {
        throw std::runtime_error("ModBaseConvLSTMV3Model expected 2 lstms");
    }

    sig_convs = register_module("sig", nn::ConvStack(sig_cvs));
    seq_convs = register_module("seq", nn::ConvStack(seq_cvs));
    merge_conv = register_module("merge", nn::ConvStack(std::vector{merge_cv}));
    lstm = register_module("lstm", nn::LSTMStack(lstms.size(), lstms.at(0).size, false));
    linear = register_module("linear", torch::nn::Linear(ll.in_size, ll.out_size));
    if (lu.has_value()) {
        upsample = register_module("upsample", nn::LinearUpsample(lu.value()));
    }
    {
        // Reserve and allocate working memory for the sequence convs
        const auto [T, C] = get_chunked_sequence_input_TC(model_config);
        wm_seq.next_NTC(batch_size, T, C, nn::TensorLayout::NTC);
        seq_convs->reserve_working_memory(wm_seq, nullptr, nn::TensorLayout::NTC);
    }
    {
        // Reserve and allocate working memory for everything else
        const auto [T, C] = get_chunked_signal_input_TC(model_config);
        wm.next_NTC(batch_size, T, C, nn::TensorLayout::NTC);
        sig_convs->reserve_working_memory(wm, nullptr, nn::TensorLayout::NTC);

        // Buffer for concatenating conv outputs before merge_conv
        const int padding = merge_cv.winlen / 2;
        const int64_t T_merge_conv = (T / model_config.modules.signal_stride()) + (2 * padding);
        wm.next_NTC(-1, T_merge_conv, merge_cv.size, nn::TensorLayout::NTC);

        std::optional<nn::TensorLayout> force_dtype = std::nullopt;
        if (nn::koi_can_use_cutlass()) {
            force_dtype = utils::get_dev_opt("modbase_i8", true)
                                  ? nn::TensorLayout::CUTLASS_TNC_I8
                                  : nn::TensorLayout::CUTLASS_TNC_F16;
            DORADO_LOG(trace, "Set modbase merge_conv CUTLASS dtype: {}",
                       to_string(force_dtype.value()));
        }

        merge_conv->reserve_working_memory(wm, nullptr, force_dtype);
        lstm->reserve_working_memory(wm);
    }
}

at::Tensor ModBaseConvLSTMV3CUDAModelImpl::forward(const at::Tensor& sigs_N1T,
                                                   const at::Tensor& seqs_NTC) {
    utils::ScopedProfileRange spr("mbv3 koi", 1);

    c10::cuda::CUDAGuard device_guard(sigs_N1T.device());

    {
        utils::ScopedProfileRange spr2("seq_copy", 2);
        const auto [Tin_seq, Cin_seq] = get_chunked_sequence_input_TC(model_config);
        if (!wm_seq.backing_tensor.defined()) {
            wm_seq.allocate_backing_tensor(seqs_NTC.device());
        }
        auto wm_seq_in = wm_seq.next_NTC(batch_size, Tin_seq, Cin_seq, nn::TensorLayout::NTC);
        wm_seq_in.index({torch::indexing::Slice()}) = seqs_NTC;
    }
    {
        utils::ScopedProfileRange spr2("sig_copy", 2);
        const auto [Tin_sig, Cin_sig] = get_chunked_signal_input_TC(model_config);
        if (!wm.backing_tensor.defined()) {
            wm.allocate_backing_tensor(sigs_N1T.device());
        }
        auto wm_in = wm.next_NTC(batch_size, Tin_sig, Cin_sig, nn::TensorLayout::NTC);
        // This transpose is free because C is 1
        wm_in.index({torch::indexing::Slice()}) = sigs_N1T.transpose(1, 2);
    }
    {
        utils::ScopedProfileRange spr2("seq_convs", 2);
        seq_convs->run_koi(wm_seq, nullptr);
    }
    {
        utils::ScopedProfileRange spr2("sig_convs", 2);
        sig_convs->run_koi(wm, nullptr);
    }

    if (wm.layout != wm_seq.layout) {
        throw std::runtime_error(
                "Signal and sequence convolution stacks must have the same output layout");
    }
    {
        // TODO: This can be removed with changes to the sequence and signal convolutions
        // Copy the output of seq_convs stored in wm_seq into wm for the merge_conv input buffer.
        utils::ScopedProfileRange spr2("merge_prep", 2);

        const auto& merge_cv = model_config.modules.merge_conv;
        auto seq_out = wm_seq.current;
        auto sig_out = wm.current;

        // If the merge conv is using CUTLASS - need to add padding
        const auto mcv_layout = merge_conv->layers.back().output_layout;
        const bool using_cutlass = mcv_layout == nn::TensorLayout::CUTLASS_TNC_F16 ||
                                   mcv_layout == nn::TensorLayout::CUTLASS_TNC_I8;

        const int padding = using_cutlass ? merge_cv.winlen / 2 : 0;
        const int T = seq_out.size(1);
        const int Tp = T + (2 * padding);
        const size_t C_sum = seq_out.size(2) + sig_out.size(2);

        // Performing torch::cat({sig, seq}, 1) but with optional padding
        auto merge_conv_in =
                wm.next_NTC(-1, Tp, C_sum, nn::TensorLayout::NTC).view({batch_size, Tp, 2, -1});
        merge_conv_in.index({torch::indexing::Slice(), torch::indexing::Slice(padding, T + padding),
                             0}) = sig_out;
        merge_conv_in.index({torch::indexing::Slice(), torch::indexing::Slice(padding, T + padding),
                             1}) = seq_out;
    }
    {
        utils::ScopedProfileRange spr2("merge_conv", 2);
        wm.is_input_to_rev_lstm = false;
        merge_conv->run_koi(wm, nullptr);
    }
    {
        utils::ScopedProfileRange spr2("lstms", 2);
        lstm->run_koi(wm, nullptr);
    }

    at::Tensor out;
    {
        utils::ScopedProfileRange spr2("linear", 2);
        out = linear->forward(wm.layout == nn::TensorLayout::CUTLASS_TNC_I8
                                      ? wm.get_current_NTC_view().to(torch::kF16) / 127.0f
                                      : wm.get_current_NTC_view());
    }
    if (model_config.modules.upsample.has_value()) {
        utils::ScopedProfileRange spr2("upsample", 2);
        out = upsample->forward(out);  // NTC -> N(sf*T)C
    }

    utils::ScopedProfileRange spr2("softmax", 2);
    return out.softmax(2).flatten(1);  // NTC -> N(TC)
}

void ModBaseConvLSTMV3CUDAModelImpl::load_state_dict(const std::vector<at::Tensor>& weights) {
    utils::load_state_dict(*this, weights);
}

std::vector<at::Tensor> ModBaseConvLSTMV3CUDAModelImpl::load_weights(
        const std::filesystem::path& dir) {
    return utils::load_tensors(
            dir, load_modbase_conv_lstm_weights(model_config.modules.upsample.has_value()));
}

#endif

}  // namespace dorado::modbase::model
