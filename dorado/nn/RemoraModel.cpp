#include "RemoraModel.h"

#include "modbase/remora_encoder.h"
#include "modbase/remora_scaler.h"
#include "modbase/remora_utils.h"
#include "utils/base_mod_utils.h"
#include "utils/module_utils.h"
#include "utils/sequence_utils.h"
#include "utils/tensor_utils.h"

#include <toml.hpp>
#include <torch/torch.h>

#include <array>
#include <stdexcept>
#include <unordered_map>

using namespace torch::nn;
using namespace torch::indexing;

struct ConvBatchNormImpl : Module {
    ConvBatchNormImpl(int size = 1,
                      int outsize = 1,
                      int k = 1,
                      int stride = 1,
                      int num_features = 1) {
        conv = register_module("conv", Conv1d(Conv1dOptions(size, outsize, k).stride(stride)));
        batch_norm = register_module("batch_norm", BatchNorm1d(num_features));
        activation = register_module("activation", SiLU());
    }

    torch::Tensor forward(torch::Tensor x) { return activation(batch_norm(conv(x))); }

    Conv1d conv{nullptr};
    BatchNorm1d batch_norm{nullptr};
    SiLU activation{nullptr};
};

TORCH_MODULE(ConvBatchNorm);

struct RemoraConvModelImpl : Module {
    RemoraConvModelImpl(int size, int kmer_len, int num_out) {
        sig_conv1 = register_module("sig_conv1", ConvBatchNorm(1, 4, 11, 1, 4));
        sig_conv2 = register_module("sig_conv2", ConvBatchNorm(4, 16, 11, 1, 16));
        sig_conv3 = register_module("sig_conv3", ConvBatchNorm(16, size, 9, 3, size));

        seq_conv1 = register_module("seq_conv1", ConvBatchNorm(kmer_len * 4, 16, 11, 1, 16));
        seq_conv2 = register_module("seq_conv2", ConvBatchNorm(16, 32, 11, 1, 32));
        seq_conv3 = register_module("seq_conv3", ConvBatchNorm(32, size, 9, 3, size));

        merge_conv1 = register_module("merge_conv1", ConvBatchNorm(size * 2, size, 5, 1, size));
        merge_conv2 = register_module("merge_conv2", ConvBatchNorm(size, size, 5, 1, size));
        merge_conv3 = register_module("merge_conv3", ConvBatchNorm(size, size, 3, 2, size));
        merge_conv4 = register_module("merge_conv4", ConvBatchNorm(size, size, 3, 2, size));

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

    void load_state_dict(const std::vector<torch::Tensor>& weights,
                         const std::vector<torch::Tensor>& buffers) {
        ::utils::load_state_dict(*this, weights, buffers);
    }

    std::vector<torch::Tensor> load_weights(const std::filesystem::path& dir) {
        return ::utils::load_tensors(dir, weight_tensors);
    }

    std::vector<torch::Tensor> load_buffers(const std::filesystem::path& dir) {
        return ::utils::load_tensors(dir, buffer_tensors);
    }

    static const std::vector<std::string> weight_tensors;
    static const std::vector<std::string> buffer_tensors;

    ConvBatchNorm sig_conv1{nullptr};
    ConvBatchNorm sig_conv2{nullptr};
    ConvBatchNorm sig_conv3{nullptr};
    ConvBatchNorm seq_conv1{nullptr};
    ConvBatchNorm seq_conv2{nullptr};
    ConvBatchNorm seq_conv3{nullptr};
    ConvBatchNorm merge_conv1{nullptr};
    ConvBatchNorm merge_conv2{nullptr};
    ConvBatchNorm merge_conv3{nullptr};
    ConvBatchNorm merge_conv4{nullptr};
    Linear linear{nullptr};
};

const std::vector<std::string> RemoraConvModelImpl::weight_tensors{
        "sig_conv1.weight.tensor",   "sig_conv1.bias.tensor",
        "sig_bn1.weight.tensor",     "sig_bn1.bias.tensor",

        "sig_conv2.weight.tensor",   "sig_conv2.bias.tensor",
        "sig_bn2.weight.tensor",     "sig_bn2.bias.tensor",

        "sig_conv3.weight.tensor",   "sig_conv3.bias.tensor",
        "sig_bn3.weight.tensor",     "sig_bn3.bias.tensor",

        "seq_conv1.weight.tensor",   "seq_conv1.bias.tensor",
        "seq_bn1.weight.tensor",     "seq_bn1.bias.tensor",

        "seq_conv2.weight.tensor",   "seq_conv2.bias.tensor",
        "seq_bn2.weight.tensor",     "seq_bn2.bias.tensor",

        "seq_conv3.weight.tensor",   "seq_conv3.bias.tensor",
        "seq_bn3.weight.tensor",     "seq_bn3.bias.tensor",

        "merge_conv1.weight.tensor", "merge_conv1.bias.tensor",
        "merge_bn1.weight.tensor",   "merge_bn1.bias.tensor",

        "merge_conv2.weight.tensor", "merge_conv2.bias.tensor",
        "merge_bn2.weight.tensor",   "merge_bn2.bias.tensor",

        "merge_conv3.weight.tensor", "merge_conv3.bias.tensor",
        "merge_bn3.weight.tensor",   "merge_bn3.bias.tensor",

        "merge_conv4.weight.tensor", "merge_conv4.bias.tensor",
        "merge_bn4.weight.tensor",   "merge_bn4.bias.tensor",

        "fc.weight.tensor",          "fc.bias.tensor",
};

const std::vector<std::string> RemoraConvModelImpl::buffer_tensors{
        "sig_bn1.running_mean.tensor",          "sig_bn1.running_var.tensor",
        "sig_bn1.num_batches_tracked.tensor",

        "sig_bn2.running_mean.tensor",          "sig_bn2.running_var.tensor",
        "sig_bn2.num_batches_tracked.tensor",

        "sig_bn3.running_mean.tensor",          "sig_bn3.running_var.tensor",
        "sig_bn3.num_batches_tracked.tensor",

        "seq_bn1.running_mean.tensor",          "seq_bn1.running_var.tensor",
        "seq_bn1.num_batches_tracked.tensor",

        "seq_bn2.running_mean.tensor",          "seq_bn2.running_var.tensor",
        "seq_bn2.num_batches_tracked.tensor",

        "seq_bn3.running_mean.tensor",          "seq_bn3.running_var.tensor",
        "seq_bn3.num_batches_tracked.tensor",

        "merge_bn1.running_mean.tensor",        "merge_bn1.running_var.tensor",
        "merge_bn1.num_batches_tracked.tensor",

        "merge_bn2.running_mean.tensor",        "merge_bn2.running_var.tensor",
        "merge_bn2.num_batches_tracked.tensor",

        "merge_bn3.running_mean.tensor",        "merge_bn3.running_var.tensor",
        "merge_bn3.num_batches_tracked.tensor",

        "merge_bn4.running_mean.tensor",        "merge_bn4.running_var.tensor",
        "merge_bn4.num_batches_tracked.tensor"};

struct RemoraConvLSTMModelImpl : Module {
    RemoraConvLSTMModelImpl(int size, int kmer_len, int num_out) {
        sig_conv1 = register_module("sig_conv1", ConvBatchNorm(1, 4, 5, 1, 4));
        sig_conv2 = register_module("sig_conv2", ConvBatchNorm(4, 16, 5, 1, 16));
        sig_conv3 = register_module("sig_conv3", ConvBatchNorm(16, size, 9, 3, size));

        seq_conv1 = register_module("seq_conv1", ConvBatchNorm(kmer_len * 4, 16, 5, 1, 16));
        seq_conv2 = register_module("seq_conv2", ConvBatchNorm(16, size, 13, 3, size));

        merge_conv1 = register_module("merge_conv1", ConvBatchNorm(size * 2, size, 5, 1, size));

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

    void load_state_dict(std::vector<torch::Tensor> weights, std::vector<torch::Tensor> buffers) {
        ::utils::load_state_dict(*this, weights, buffers);
    }

    std::vector<torch::Tensor> load_weights(const std::filesystem::path& dir) {
        return ::utils::load_tensors(dir, weight_tensors);
    }

    std::vector<torch::Tensor> load_buffers(const std::filesystem::path& dir) {
        return ::utils::load_tensors(dir, buffer_tensors);
    }

    static const std::vector<std::string> weight_tensors;
    static const std::vector<std::string> buffer_tensors;

    ConvBatchNorm sig_conv1{nullptr};
    ConvBatchNorm sig_conv2{nullptr};
    ConvBatchNorm sig_conv3{nullptr};
    ConvBatchNorm seq_conv1{nullptr};
    ConvBatchNorm seq_conv2{nullptr};
    ConvBatchNorm merge_conv1{nullptr};

    LSTM lstm1{nullptr};
    LSTM lstm2{nullptr};

    Linear linear{nullptr};
    SiLU activation{nullptr};
};

const std::vector<std::string> RemoraConvLSTMModelImpl::weight_tensors{
        "sig_conv1.weight.tensor",   "sig_conv1.bias.tensor",
        "sig_bn1.weight.tensor",     "sig_bn1.bias.tensor",

        "sig_conv2.weight.tensor",   "sig_conv2.bias.tensor",
        "sig_bn2.weight.tensor",     "sig_bn2.bias.tensor",

        "sig_conv3.weight.tensor",   "sig_conv3.bias.tensor",
        "sig_bn3.weight.tensor",     "sig_bn3.bias.tensor",

        "seq_conv1.weight.tensor",   "seq_conv1.bias.tensor",
        "seq_bn1.weight.tensor",     "seq_bn1.bias.tensor",

        "seq_conv2.weight.tensor",   "seq_conv2.bias.tensor",
        "seq_bn2.weight.tensor",     "seq_bn2.bias.tensor",

        "merge_conv1.weight.tensor", "merge_conv1.bias.tensor",
        "merge_bn.weight.tensor",    "merge_bn.bias.tensor",

        "lstm1.weight_ih_l0.tensor", "lstm1.weight_hh_l0.tensor",
        "lstm1.bias_ih_l0.tensor",   "lstm1.bias_hh_l0.tensor",

        "lstm2.weight_ih_l0.tensor", "lstm2.weight_hh_l0.tensor",
        "lstm2.bias_ih_l0.tensor",   "lstm2.bias_hh_l0.tensor",

        "fc.weight.tensor",          "fc.bias.tensor",
};

const std::vector<std::string> RemoraConvLSTMModelImpl::buffer_tensors{
        "sig_bn1.running_mean.tensor",        "sig_bn1.running_var.tensor",
        "sig_bn1.num_batches_tracked.tensor",

        "sig_bn2.running_mean.tensor",        "sig_bn2.running_var.tensor",
        "sig_bn2.num_batches_tracked.tensor",

        "sig_bn3.running_mean.tensor",        "sig_bn3.running_var.tensor",
        "sig_bn3.num_batches_tracked.tensor",

        "seq_bn1.running_mean.tensor",        "seq_bn1.running_var.tensor",
        "seq_bn1.num_batches_tracked.tensor",

        "seq_bn2.running_mean.tensor",        "seq_bn2.running_var.tensor",
        "seq_bn2.num_batches_tracked.tensor",

        "merge_bn.running_mean.tensor",       "merge_bn.running_var.tensor",
        "merge_bn.num_batches_tracked.tensor"};

TORCH_MODULE(RemoraConvModel);
TORCH_MODULE(RemoraConvLSTMModel);

namespace {
template <class Model>
ModuleHolder<AnyModule> populate_model(Model&& model,
                                       const std::filesystem::path& path,
                                       torch::TensorOptions options) {
    auto state_dict = model->load_weights(path);
    auto state_buffers = model->load_buffers(path);
    model->load_state_dict(state_dict, state_buffers);
    model->to(options.dtype_opt().value().toScalarType());
    model->to(options.device_opt().value());
    model->eval();

    auto module = AnyModule(model);
    auto holder = ModuleHolder<AnyModule>(module);
    return holder;
}
}  // namespace

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
        auto model = RemoraConvLSTMModel(size, kmer_len, num_out);
        return populate_model(model, model_path, options);
    }

    if (model_type == "conv_only") {
        auto model = RemoraConvModel(size, kmer_len, num_out);
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
    m_options = torch::TensorOptions().dtype(dtype).device(device == "metal" ? "cpu" : device);
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
                    ::utils::load_tensors(model_path, {"refine_kmer_levels.tensor"})[0]
                            .contiguous();
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

    m_input_sigs = torch::zeros({batch_size, 1, sig_len},
                                torch::TensorOptions().dtype(dtype).device(torch::kCPU));
    m_input_seqs = torch::zeros({batch_size, RemoraUtils::NUM_BASES * kmer_len, sig_len},
                                torch::TensorOptions().dtype(dtype).device(torch::kCPU));

    auto context_samples = (m_params.context_before + m_params.context_after);
    // One-hot encodes the kmer at each signal step for input into the network
    m_encoder = std::make_unique<RemoraEncoder>(block_stride, context_samples,
                                                m_params.bases_before, m_params.bases_after);
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

std::pair<torch::Tensor, std::vector<size_t>> RemoraCaller::call(
        torch::Tensor signal,
        const std::string& seq,
        const std::vector<uint8_t>& moves) {
    // TODO: could just do this on the candidates instead of the entire read
    m_encoder->encode_remora_data(moves, seq);
    auto context_hits = get_motif_hits(seq);
    auto counter = 0;
    auto index = 0;
    auto kmer_len = m_params.bases_before + m_params.bases_after + 1;

    // Network outputs
    auto scores = torch::empty({static_cast<int64_t>(context_hits.size()),
                                static_cast<int64_t>(m_params.base_mod_count + 1)});

    for (auto context_hit : context_hits) {
        auto slice = m_encoder->get_context(context_hit);
        size_t first_sample_source = slice.first_sample;
        size_t last_sample_source = first_sample_source + slice.num_samples;
        size_t first_sample_dest = slice.lead_samples_needed;
        size_t last_sample_dest = first_sample_dest + slice.num_samples;
        size_t tail_samples_needed = slice.tail_samples_needed;
        if (last_sample_source > signal.size(0)) {
            size_t overrun = last_sample_source - signal.size(0);
            tail_samples_needed += overrun;
            last_sample_dest -= overrun;
            last_sample_source -= overrun;
        }

        auto input_signal = signal.index({Slice(first_sample_source, last_sample_source)});
        m_input_sigs.index_put_({counter, 0, Slice(first_sample_dest, last_sample_dest)},
                                input_signal);

        if (slice.lead_samples_needed > 0) {
            m_input_sigs.index_put_({counter, 0, Slice(None, first_sample_dest)}, 0);
        }
        if (tail_samples_needed > 0) {
            m_input_sigs.index_put_({counter, 0, Slice(last_sample_dest, None)}, 0);
        }

        auto context_samples = (m_params.context_before + m_params.context_after);
        m_input_seqs.index_put_(
                {counter}, torch::from_blob(slice.data.data(), {(int64_t)context_samples,
                                                                kmer_len * RemoraUtils::NUM_BASES})
                                   .transpose(0, 1));
        if (++counter == m_batch_size) {
            torch::InferenceMode guard;
            counter = 0;
            auto output = m_module->forward(m_input_sigs.to(m_options.device_opt().value()),
                                            m_input_seqs.to(m_options.device_opt().value()));
            scores.index_put_({Slice(index, index + m_batch_size), Slice(0, output.size(1))},
                              output.to(torch::kCPU));
            index += m_batch_size;
        }
    }

    if (counter != 0) {
        torch::InferenceMode guard;
        auto output = m_module->forward(m_input_sigs.index({Slice(0, counter), Slice(), Slice()})
                                                .to(m_options.device_opt().value()),
                                        m_input_seqs.index({Slice(0, counter), Slice(), Slice()})
                                                .to(m_options.device_opt().value()));
        scores.index_put_({Slice(index, index + counter), Slice(0, output.size(1))},
                          output.to(torch::kCPU));
    }
    return {scores, context_hits};
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

RemoraRunner::RemoraRunner(const std::vector<std::filesystem::path>& model_paths,
                           const std::string& device,
                           size_t block_stride,
                           int batch_size)
        : m_num_states(4),              // The 4 canonical bases.
          m_block_stride(block_stride)  // The stride of the associated basecall model
{
    struct Info {
        std::vector<std::string> long_names;
        std::string alphabet;
        std::string motif;
        int motif_offset;
    };

    std::string allowed_bases = "ACGT";
    std::array<Info, 4> model_info;
    for (int b = 0; b < 4; ++b) {
        model_info[b].alphabet = allowed_bases[b];
    }

    std::array<size_t, 4> base_counts = {1, 1, 1, 1};
    std::array<bool, 4> base_used = {false, false, false, false};
    for (const auto& model : model_paths) {
        auto caller = std::make_shared<RemoraCaller>(model, device, batch_size, block_stride);
        auto& params = caller->params();

        auto base = params.motif[params.motif_offset];
        if (allowed_bases.find(base) == std::string::npos) {
            throw std::runtime_error("Invalid base in remora model metadata.");
        }
        auto& map_entry = model_info[RemoraUtils::BASE_IDS[base]];
        map_entry.long_names = params.mod_long_names;
        map_entry.alphabet += params.mod_bases;
        map_entry.motif = params.motif;
        map_entry.motif_offset = params.motif_offset;

        base_counts[RemoraUtils::BASE_IDS[base]] = params.base_mod_count + 1;
        m_num_states += params.base_mod_count;
        m_callers.push_back(caller);
    }

    std::string long_names, alphabet;
    ::utils::BaseModContext context_handler;
    for (const auto& info : model_info) {
        for (const auto& name : info.long_names) {
            if (!long_names.empty())
                long_names += ' ';
            long_names += name;
        }
        alphabet += info.alphabet;
        if (!info.motif.empty()) {
            context_handler.set_context(info.motif, size_t(info.motif_offset));
        }
    }

    m_base_mod_info = std::make_shared<BaseModInfo>(alphabet, long_names, context_handler.encode());

    m_base_prob_offsets[0] = 0;
    m_base_prob_offsets[1] = base_counts[0];
    m_base_prob_offsets[2] = base_counts[0] + base_counts[1];
    m_base_prob_offsets[3] = base_counts[0] + base_counts[1] + base_counts[2];
}

std::vector<uint8_t> RemoraRunner::run(torch::Tensor signal,
                                       const std::string& seq,
                                       const std::vector<uint8_t>& moves) {
    // TODO: this is probably quite expensive
    // encode the modbase alphabet probabilities at each base
    std::vector<uint8_t> base_mod_probs(seq.size() * m_num_states, 0);

    for (size_t i = 0; i < seq.size(); ++i) {
        // Initialize for what corresponds to 100% canonical base for each position.
        int base_id = RemoraUtils::BASE_IDS[seq[i]];
        if (base_id < 0) {
            throw std::runtime_error("Invalid character in sequence.");
        }
        base_mod_probs[i * m_num_states + m_base_prob_offsets[base_id]] = 1.0f;
    }

    std::vector<int> sequence_ints = ::utils::sequence_to_ints(seq);

    // Convert the move table to a series of indicies in the signal corresponding to the position of the 1s in the move table
    std::vector<uint64_t> seq_to_sig_map =
            ::utils::moves_to_map(moves, m_block_stride, signal.size(0));

    // each caller will have different parameters
    for (auto& caller : m_callers) {
        auto scaled_signal = caller->scale_signal(signal, sequence_ints, seq_to_sig_map);
        // The scores from the RNN should be a MxN tensor,
        // where M is the number of context hits and N is the number of modifications + 1.
        auto [scores, context_hits] = caller->call(scaled_signal, seq, moves);
        for (size_t i = 0; i < context_hits.size(); ++i) {
            int64_t result_pos = context_hits[i];
            int64_t offset = m_base_prob_offsets[RemoraUtils::BASE_IDS[seq[context_hits[i]]]];
            for (int j = 0; j < scores.size(1); ++j) {
                base_mod_probs[m_num_states * result_pos + offset + j] = uint8_t(std::min(
                        std::floor(scores.index({(int64_t)i, j}).item().toFloat() * 256), 255.0f));
            }
        }
    }

    return base_mod_probs;
}
