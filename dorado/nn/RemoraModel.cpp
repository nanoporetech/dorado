#include "RemoraModel.h"

#include "modbase/remora_encoder.h"
#include "modbase/remora_scaler.h"
#include "modbase/remora_utils.h"
#include "utils/base_mod_utils.h"
#include "utils/module_utils.h"
#include "utils/sequence_utils.h"
#include "utils/tensor_utils.h"

#ifndef __APPLE__
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <nvtx3/nvtx3.hpp>
#include <toml.hpp>
#include <torch/torch.h>

#include <array>
#include <stdexcept>
#include <unordered_map>

using namespace torch::nn;
using namespace torch::indexing;

namespace {
template <class Model>
ModuleHolder<AnyModule> populate_model(Model&& model,
                                       const std::filesystem::path& path,
                                       torch::TensorOptions options) {
    auto state_dict = model->load_weights(path);
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

struct UnpaddedConvolutionImpl : Module {
    UnpaddedConvolutionImpl(int size = 1, int outsize = 1, int k = 1, int stride = 1) {
        conv = register_module("conv", Conv1d(Conv1dOptions(size, outsize, k).stride(stride)));
        activation = register_module("activation", SiLU());
    }

    torch::Tensor forward(torch::Tensor x) { return activation(conv(x)); }

    Conv1d conv{nullptr};
    SiLU activation{nullptr};
};

TORCH_MODULE(UnpaddedConvolution);

struct RemoraConvModelImpl : Module {
    RemoraConvModelImpl(int size, int kmer_len, int num_out) {
        sig_conv1 = register_module("sig_conv1", UnpaddedConvolution(1, 4, 11, 1));
        sig_conv2 = register_module("sig_conv2", UnpaddedConvolution(4, 16, 11, 1));
        sig_conv3 = register_module("sig_conv3", UnpaddedConvolution(16, size, 9, 3));

        seq_conv1 = register_module("seq_conv1", UnpaddedConvolution(kmer_len * 4, 16, 11, 1));
        seq_conv2 = register_module("seq_conv2", UnpaddedConvolution(16, 32, 11, 1));
        seq_conv3 = register_module("seq_conv3", UnpaddedConvolution(32, size, 9, 3));

        merge_conv1 = register_module("merge_conv1", UnpaddedConvolution(size * 2, size, 5, 1));
        merge_conv2 = register_module("merge_conv2", UnpaddedConvolution(size, size, 5, 1));
        merge_conv3 = register_module("merge_conv3", UnpaddedConvolution(size, size, 3, 2));
        merge_conv4 = register_module("merge_conv4", UnpaddedConvolution(size, size, 3, 2));

        linear = register_module("linear", Linear(size * 3, num_out));
    }

    torch::Tensor forward(torch::Tensor sigs, torch::Tensor seqs) {
        sigs = sig_conv1(sigs);
        sigs = sig_conv2(sigs);
        sigs = sig_conv3(sigs);

        seqs = seq_conv1(seqs);
        seqs = seq_conv2(seqs);
        seqs = seq_conv3(seqs);

        auto z = torch::cat({sigs, seqs}, 1);

        z = merge_conv1(z);
        z = merge_conv2(z);
        z = merge_conv3(z);
        z = merge_conv4(z);

        z = z.flatten(1);
        z = linear(z);

        z = z.softmax(1);

        return z;
    }

    void load_state_dict(const std::vector<torch::Tensor>& weights) {
        utils::load_state_dict(*this, weights);
    }

    std::vector<torch::Tensor> load_weights(const std::filesystem::path& dir) {
        return utils::load_tensors(dir, weight_tensors);
    }

    static const std::vector<std::string> weight_tensors;

    UnpaddedConvolution sig_conv1{nullptr};
    UnpaddedConvolution sig_conv2{nullptr};
    UnpaddedConvolution sig_conv3{nullptr};
    UnpaddedConvolution seq_conv1{nullptr};
    UnpaddedConvolution seq_conv2{nullptr};
    UnpaddedConvolution seq_conv3{nullptr};
    UnpaddedConvolution merge_conv1{nullptr};
    UnpaddedConvolution merge_conv2{nullptr};
    UnpaddedConvolution merge_conv3{nullptr};
    UnpaddedConvolution merge_conv4{nullptr};
    Linear linear{nullptr};
};

const std::vector<std::string> RemoraConvModelImpl::weight_tensors{
        "sig_conv1.weight.tensor",   "sig_conv1.bias.tensor",
        "sig_conv2.weight.tensor",   "sig_conv2.bias.tensor",
        "sig_conv3.weight.tensor",   "sig_conv3.bias.tensor",

        "seq_conv1.weight.tensor",   "seq_conv1.bias.tensor",
        "seq_conv2.weight.tensor",   "seq_conv2.bias.tensor",
        "seq_conv3.weight.tensor",   "seq_conv3.bias.tensor",

        "merge_conv1.weight.tensor", "merge_conv1.bias.tensor",
        "merge_conv2.weight.tensor", "merge_conv2.bias.tensor",
        "merge_conv3.weight.tensor", "merge_conv3.bias.tensor",
        "merge_conv4.weight.tensor", "merge_conv4.bias.tensor",

        "fc.weight.tensor",          "fc.bias.tensor",
};

struct RemoraConvLSTMModelImpl : Module {
    RemoraConvLSTMModelImpl(int size, int kmer_len, int num_out) {
        sig_conv1 = register_module("sig_conv1", UnpaddedConvolution(1, 4, 5, 1));
        sig_conv2 = register_module("sig_conv2", UnpaddedConvolution(4, 16, 5, 1));
        sig_conv3 = register_module("sig_conv3", UnpaddedConvolution(16, size, 9, 3));

        seq_conv1 = register_module("seq_conv1", UnpaddedConvolution(kmer_len * 4, 16, 5, 1));
        seq_conv2 = register_module("seq_conv2", UnpaddedConvolution(16, size, 13, 3));

        merge_conv1 = register_module("merge_conv1", UnpaddedConvolution(size * 2, size, 5, 1));

        lstm1 = register_module("lstm1", LSTM(LSTMOptions(size, size)));
        lstm2 = register_module("lstm2", LSTM(LSTMOptions(size, size)));

        linear = register_module("linear", Linear(size, num_out));

        activation = register_module("activation", SiLU());
    }

    torch::Tensor forward(torch::Tensor sigs, torch::Tensor seqs) {
        sigs = sig_conv1(sigs);
        sigs = sig_conv2(sigs);
        sigs = sig_conv3(sigs);

        seqs = seq_conv1(seqs);
        seqs = seq_conv2(seqs);

        auto z = torch::cat({sigs, seqs}, 1);
        z = merge_conv1(z);
        z = z.permute({2, 0, 1});

        auto [z1, h1] = lstm1(z);
        z1 = activation(z1);

        z1 = z1.flip(0);
        auto [z2, h2] = lstm2(z1);
        z2 = activation(z2);
        z2 = z2.flip(0);

        z = z2.index({-1}).permute({0, 1});
        z = linear(z);
        z = z.softmax(1);

        return z;
    }

    void load_state_dict(std::vector<torch::Tensor> weights) {
        utils::load_state_dict(*this, weights);
    }

    std::vector<torch::Tensor> load_weights(const std::filesystem::path& dir) {
        return utils::load_tensors(dir, weight_tensors);
    }

    static const std::vector<std::string> weight_tensors;

    UnpaddedConvolution sig_conv1{nullptr};
    UnpaddedConvolution sig_conv2{nullptr};
    UnpaddedConvolution sig_conv3{nullptr};
    UnpaddedConvolution seq_conv1{nullptr};
    UnpaddedConvolution seq_conv2{nullptr};
    UnpaddedConvolution merge_conv1{nullptr};

    LSTM lstm1{nullptr};
    LSTM lstm2{nullptr};

    Linear linear{nullptr};
    SiLU activation{nullptr};
};

const std::vector<std::string> RemoraConvLSTMModelImpl::weight_tensors{
        "sig_conv1.weight.tensor",   "sig_conv1.bias.tensor",
        "sig_conv2.weight.tensor",   "sig_conv2.bias.tensor",
        "sig_conv3.weight.tensor",   "sig_conv3.bias.tensor",

        "seq_conv1.weight.tensor",   "seq_conv1.bias.tensor",
        "seq_conv2.weight.tensor",   "seq_conv2.bias.tensor",

        "merge_conv1.weight.tensor", "merge_conv1.bias.tensor",

        "lstm1.weight_ih_l0.tensor", "lstm1.weight_hh_l0.tensor",
        "lstm1.bias_ih_l0.tensor",   "lstm1.bias_hh_l0.tensor",

        "lstm2.weight_ih_l0.tensor", "lstm2.weight_hh_l0.tensor",
        "lstm2.bias_ih_l0.tensor",   "lstm2.bias_hh_l0.tensor",

        "fc.weight.tensor",          "fc.bias.tensor",
};

TORCH_MODULE(RemoraConvModel);
TORCH_MODULE(RemoraConvLSTMModel);

}  // namespace nn

ModuleHolder<AnyModule> load_remora_model(const std::filesystem::path& model_path,
                                          torch::TensorOptions options) {
    auto config = toml::parse(model_path / "config.toml");

    const auto& general_params = toml::find(config, "general");
    const auto model_type = toml::find<std::string>(general_params, "model");

    const auto& model_params = toml::find(config, "model_params");
    const auto size = toml::find<int>(model_params, "size");
    const auto kmer_len = toml::find<int>(model_params, "kmer_len");
    const auto num_out = toml::find<int>(model_params, "num_out");

    if (model_type == "conv_lstm") {
        auto model = nn::RemoraConvLSTMModel(size, kmer_len, num_out);
        return populate_model(model, model_path, options);
    }

    if (model_type == "conv_only") {
        auto model = nn::RemoraConvModel(size, kmer_len, num_out);
        return populate_model(model, model_path, options);
    }

    throw std::runtime_error("Unknown model type in config file.");
}

RemoraCaller::RemoraCaller(const std::filesystem::path& model_path,
                           const std::string& device,
                           int batch_size,
                           size_t block_stride)
        : m_batch_size(batch_size) {
    // no metal implementation yet, force to cpu
    if (device == "metal" || device == "cpu") {
        // no slow_conv2d_cpu for type Half, need to use float32
        m_options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
    } else {
        m_options = torch::TensorOptions().device(device).dtype(torch::kFloat16);
    }

#ifndef __APPLE__
    if (m_options.device().is_cuda()) {
        m_stream = c10::cuda::getStreamFromPool(false, m_options.device().index());
    }
#endif
    m_module = load_remora_model(model_path, m_options);

    auto config = toml::parse(model_path / "config.toml");
    const auto& params = toml::find(config, "modbases");
    m_params.motif = toml::find<std::string>(params, "motif");
    m_params.motif_offset = toml::find<int>(params, "motif_offset");

    m_params.mod_bases = toml::find<std::string>(params, "mod_bases");
    for (size_t i = 0; i < m_params.mod_bases.size(); ++i) {
        m_params.mod_long_names.push_back(
                toml::find<std::string>(params, "mod_long_names_" + std::to_string(i)));
    }
    m_params.base_mod_count = m_params.mod_long_names.size();

    m_params.context_before = toml::find<int>(params, "chunk_context_0");
    m_params.context_after = toml::find<int>(params, "chunk_context_1");
    m_params.bases_before = toml::find<int>(params, "kmer_context_bases_0");
    m_params.bases_after = toml::find<int>(params, "kmer_context_bases_1");
    m_params.offset = toml::find<int>(params, "offset");

    try {
        // these may not exist if we convert older models
        const auto& refinement_params = toml::find(config, "refinement");
        m_params.refine_do_rough_rescale =
                (toml::find<int>(refinement_params, "refine_do_rough_rescale") == 1);
        if (m_params.refine_do_rough_rescale) {
            m_params.refine_kmer_center_idx =
                    toml::find<int>(refinement_params, "refine_kmer_center_idx");

            auto kmer_levels_tensor =
                    utils::load_tensors(model_path, {"refine_kmer_levels.tensor"})[0].contiguous();
            std::copy(kmer_levels_tensor.data_ptr<float>(),
                      kmer_levels_tensor.data_ptr<float>() + kmer_levels_tensor.numel(),
                      std::back_inserter(m_params.refine_kmer_levels));
            m_params.refine_kmer_len = static_cast<size_t>(
                    std::round(std::log(m_params.refine_kmer_levels.size()) / std::log(4)));
        }

    } catch (const std::out_of_range& ex) {
        // no refinement parameters
        m_params.refine_do_rough_rescale = false;
    }

    auto sig_len = static_cast<int64_t>(m_params.context_before + m_params.context_after);
    auto kmer_len = m_params.bases_after + m_params.bases_before + 1;

    auto input_options = torch::TensorOptions()
                                 .pinned_memory(m_options.device().is_cuda())
                                 .dtype(m_options.dtype())
                                 .device(torch::kCPU);

    m_input_sigs = torch::empty({batch_size, 1, sig_len}, input_options);
    m_input_seqs =
            torch::empty({batch_size, RemoraUtils::NUM_BASES * kmer_len, sig_len}, input_options);

    if (m_params.refine_do_rough_rescale) {
        m_scaler = std::make_unique<RemoraScaler>(m_params.refine_kmer_levels,
                                                  m_params.refine_kmer_len,
                                                  m_params.refine_kmer_center_idx);
    }
}

std::vector<size_t> RemoraCaller::get_motif_hits(const std::string& seq) const {
    std::vector<size_t> context_hits;
    const auto& motif = m_params.motif;
    const auto motif_offset = m_params.motif_offset;
    size_t kmer_len = motif.size();
    size_t search_pos = 0;
    while (search_pos < seq.size() - kmer_len + 1) {
        search_pos = seq.find(motif, search_pos);
        if (search_pos != std::string::npos) {
            context_hits.push_back(search_pos + motif_offset);
            ++search_pos;
        }
    }
    return context_hits;
}

torch::Tensor RemoraCaller::scale_signal(torch::Tensor signal,
                                         const std::vector<int>& seq_ints,
                                         const std::vector<uint64_t>& seq_to_sig_map) const {
    if (!m_scaler) {
        return signal;
    }

    auto levels = m_scaler->extract_levels(seq_ints);

    // generate the signal values at the centre of each base, create the nx5% quantiles (sorted)
    // and perform a linear regression against the expected kmer levels to generate a new shift and scale
    auto [offset, scale] = m_scaler->rescale(signal, seq_to_sig_map, levels);
    auto scaled_signal = signal * scale + offset;
    return scaled_signal;
}

void RemoraCaller::accept_chunk(int num_chunks,
                                at::Tensor signal,
                                const std::vector<float>& kmers) {
    m_input_sigs.index_put_({num_chunks, 0}, signal);

    auto sig_len = static_cast<int64_t>(m_params.context_before + m_params.context_after);
    auto kmer_len = m_params.bases_after + m_params.bases_before + 1;
    auto slice = torch::from_blob(const_cast<float*>(kmers.data()),
                                  {kmer_len * RemoraUtils::NUM_BASES, sig_len});
    m_input_seqs.index_put_({num_chunks}, slice);
}

torch::Tensor RemoraCaller::call_chunks(int num_chunks) {
    nvtx3::scoped_range loop{"remora_nn"};

    torch::InferenceMode guard;

#ifndef __APPLE__
    // If m_stream is set, sets the current stream to m_stream, and the current device to the device associated
    // with the stream. Resets both to their prior state on destruction
    c10::cuda::OptionalCUDAStreamGuard stream_guard(m_stream);
#endif
    auto scores = m_module->forward(m_input_sigs.to(m_options.device()),
                                    m_input_seqs.to(m_options.device()));

    return scores.to(torch::kCPU);
}

}  // namespace dorado
