#include "RemoraModel.h"

#include "../utils/base64_utils.h"
#include "../utils/math_utils.h"
#include "../utils/module_utils.h"
#include "../utils/tensor_utils.h"

#include <toml.hpp>
#include <torch/torch.h>

#include <array>
#include <stdexcept>

using namespace torch::nn;
using namespace torch::indexing;

namespace {
constexpr int NUM_BASES = 4;
}

struct ConvBatchNormImpl : Module {
    ConvBatchNormImpl(int size = 1,
                      int outsize = 1,
                      int k = 1,
                      int stride = 1,
                      int num_features = 1) {
        conv = register_module(
                "conv", Conv1d(Conv1dOptions(size, outsize, k).stride(stride).padding(k / 2)));
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

    void load_state_dict(const std::vector<torch::Tensor>& weights) {
        ::utils::load_state_dict(*this, weights);
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

    std::vector<torch::Tensor> load_weights(const std::string& dir) {
        auto weights = std::vector<torch::Tensor>();
        auto tensors = std::vector<std::string>{
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

        return ::utils::load_weights(dir, tensors);
    }

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

    void load_state_dict(std::vector<torch::Tensor> weights) {
        ::utils::load_state_dict(*this, weights);
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

        z = z2.index({-1});
        z = linear(z);

        return z;
    }

    std::vector<torch::Tensor> load_weights(const std::string& dir) {
        auto weights = std::vector<torch::Tensor>();
        auto tensors = std::vector<std::string>{
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

        return ::utils::load_weights(dir, tensors);
    }

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
    Softmax softmax{nullptr};
};
TORCH_MODULE(RemoraConvModel);
TORCH_MODULE(RemoraConvLSTMModel);

ModuleHolder<AnyModule> load_remora_model(const std::string& path, torch::TensorOptions options) {
    auto config = toml::parse(path + "/config.toml");

    const auto& model_params = toml::find(config, "model_params");
    const auto size = toml::find<int>(model_params, "size");
    const auto kmer_len = toml::find<int>(model_params, "kmer_len");
    const auto num_out = toml::find<int>(model_params, "num_out");

    // TODO: detect the correct model!
    auto model = RemoraConvLSTMModel(size, kmer_len, num_out);
    auto state_dict = model->load_weights(path);
    model->load_state_dict(state_dict);
    model->to(options.dtype_opt().value().toScalarType());
    model->to(options.device_opt().value());
    model->eval();

    auto module = AnyModule(model);
    auto holder = ModuleHolder<AnyModule>(module);

    return holder;
}

RemoraCaller::RemoraCaller(const std::string& model, const std::string& device, int batch_size)
        : m_batch_size(batch_size) {
    // no metal implementation yet, force to cpu
    m_options = torch::TensorOptions().dtype(dtype).device(device == "metal" ? "cpu" : device);
    m_module = load_remora_model(model, m_options);

    auto config = toml::parse(model + "/config.toml");
    const auto& params = toml::find(config, "modbases");
    m_params.num_motifs = toml::find<int>(params, "num_motifs");
    for (auto i = 0; i < m_params.num_motifs; ++i) {
        m_params.motifs.push_back(toml::find<std::string>(params, "motif_" + std::to_string(i)));
        m_params.motif_offsets.push_back(
                toml::find<int>(params, "motif_offset_" + std::to_string(i)));
    }

    m_params.mod_bases = toml::find<std::string>(params, "mod_bases");
    for (size_t i = 0; i < m_params.mod_bases.size(); ++i) {
        m_params.mod_long_names.push_back(
                toml::find<std::string>(params, "mod_long_names_" + std::to_string(i)));
    }
    m_params.base_mod_counts = m_params.mod_long_names.size();

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
            const auto refine_kmer_levels_base64 =
                    toml::find<std::string>(refinement_params, "refine_kmer_levels_binary");
            ::utils::decode_base64(refine_kmer_levels_base64, m_params.refine_kmer_levels);
        }

    } catch (const std::out_of_range& ex) {
        // no refinement parameters
        m_params.refine_do_rough_rescale = false;
    }

    auto sig_len = static_cast<int64_t>(m_params.context_before + m_params.context_after);
    auto kmer_len = m_params.bases_after + m_params.bases_before + 1;

    m_input_sigs = torch::zeros({batch_size, 1, sig_len},
                                torch::TensorOptions().dtype(dtype).device(torch::kCPU));
    m_input_seqs = torch::zeros({batch_size, NUM_BASES * kmer_len, sig_len},
                                torch::TensorOptions().dtype(dtype).device(torch::kCPU));
}

std::vector<size_t> RemoraCaller::get_motif_hits(const std::string& seq) const {
    std::vector<size_t> context_hits;
    for (int i = 0; i < m_params.num_motifs; ++i) {
        const auto& motif = m_params.motifs[i];
        const auto motif_offset = m_params.motif_offsets[i];
        size_t kmer_len = motif.size();
        size_t search_pos = 0;
        while (search_pos < seq.size() - kmer_len + 1) {
            search_pos = seq.find(motif, search_pos);
            if (search_pos != std::string::npos) {
                context_hits.push_back(search_pos + motif_offset);
                ++search_pos;
            }
        }
    }
    return context_hits;
}

void RemoraCaller::call(torch::Tensor signal,
                        const std::string& seq,
                        const std::vector<uint8_t>& moves) {
    auto block_stride = ::utils::div_round_closest(signal.size(0), moves.size());
    RemoraEncoder encoder(block_stride,
                          (m_params.context_before + m_params.context_after) / block_stride,
                          m_params.bases_before, m_params.bases_after);
    encoder.encode_remora_data(moves, seq);
    auto context_hits = get_motif_hits(seq);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto kmer_len = m_params.bases_after + m_params.bases_before + 1;
    auto sig_len = static_cast<int64_t>(m_params.context_before + m_params.context_after);

    auto counter = 0;
    auto index = 0;
    auto scores = torch::empty({static_cast<int64_t>(context_hits.size()),
                                static_cast<int64_t>(m_params.base_mod_counts + 1)});

    for (auto context_hit : context_hits) {
        auto slice = encoder.get_context(context_hit);
        size_t first_sample_source = slice.first_sample;
        size_t last_sample_source = first_sample_source + slice.num_samples;
        size_t first_sample_dest = slice.lead_samples_needed;
        size_t last_sample_dest = first_sample_dest + slice.num_samples;
        size_t total_samples =
                slice.lead_samples_needed + slice.num_samples + slice.tail_samples_needed;
        auto input_signal = signal.index({Slice(first_sample_source, last_sample_source)});
        m_input_sigs.index_put_({counter, 0, Slice(first_sample_dest, last_sample_dest)},
                                input_signal);

        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        torch::Tensor encoded_kmers = torch::empty({4 * kmer_len, sig_len}, options);
        std::copy_n(slice.data, slice.size, encoded_kmers.contiguous().data_ptr<float>());
        m_input_seqs.index_put_({counter}, encoded_kmers);
        if (++counter == m_batch_size) {
            counter = 0;
            auto output = m_module->forward(m_input_sigs.to(m_options.device_opt().value()),
                                            m_input_seqs.to(m_options.device_opt().value()));
            scores.index_put_({Slice(index, index + m_batch_size), Slice(0, output.size(1))},
                              output.to(torch::kCPU));
            index += m_batch_size;
        }
    }

    if (counter != 0) {
        auto output = m_module->forward(m_input_sigs.index({Slice(0, counter), Slice(), Slice()})
                                                .to(m_options.device_opt().value()),
                                        m_input_seqs.index({Slice(0, counter), Slice(), Slice()})
                                                .to(m_options.device_opt().value()));
        scores.index_put_({Slice(index, index + counter), Slice(0, output.size(1))},
                          output.to(torch::kCPU));
    }
}

RemoraRunner::RemoraRunner(const std::vector<std::string>& model_paths, const std::string& device) {
    for (const auto& model : model_paths) {
        m_callers.push_back(std::make_shared<RemoraCaller>(model, device));
    }
}

void RemoraRunner::run(torch::Tensor signal,
                       const std::string& seq,
                       const std::vector<uint8_t>& moves) {
    // each caller will have different parameters
    for (auto& caller : m_callers) {
        caller->call(signal, seq, moves);
    }
}

RemoraEncoder::RemoraEncoder(size_t block_stride,
                             size_t context_blocks,
                             int bases_before,
                             int bases_after)
        : m_base_mapping(255, -1),
          m_bases_before(bases_before),
          m_kmer_len(bases_before + bases_after + 1),
          m_block_stride(int(block_stride)),
          m_context_blocks(int(context_blocks)),
          m_seq_len(0),
          m_signal_len(0) {
    m_padding = m_context_blocks / 2;
    int padding_for_bases_before = (m_kmer_len - 1 - bases_before) * int(block_stride);
    int padding_for_bases_after = (m_kmer_len - 1 - bases_after) * int(block_stride);
    int padding_for_bases = std::max(padding_for_bases_before, padding_for_bases_after);
    m_padding = std::max(padding_for_bases, m_padding);
    m_base_mapping['A'] = 0;
    m_base_mapping['C'] = 1;
    m_base_mapping['G'] = 2;
    m_base_mapping['T'] = 3;
}

void RemoraEncoder::encode_remora_data(const std::vector<uint8_t>& moves,
                                       const std::string& sequence) {
    // This code assumes that the first move value will always be 1. It also assumes that moves is only ever 0 or 1.
    m_seq_len = int(sequence.size());
    m_signal_len = int(moves.size()) * m_block_stride;
    int padded_signal_len = m_signal_len + m_block_stride * m_padding * 2;
    int encoded_data_size = padded_signal_len * m_kmer_len * NUM_BASES;
    m_sample_offsets.clear();
    m_sample_offsets.reserve(moves.size());
    m_encoded_data.resize(encoded_data_size);

    // Note that upon initialization, encoded_data is all zeros, which corresponds to "N" characters.

    // First we need to find out which sample each base corresponds to, and make sure the moves vector is consistent
    // with the sequence length.
    int base_count = 0;
    for (int i = 0; i < int(moves.size()); ++i) {
        if (i == 0 || moves[i] == 1) {
            m_sample_offsets.push_back(i * m_block_stride);
            ++base_count;
        }
    }
    if (base_count > m_seq_len) {
        throw std::runtime_error("Movement table indicates more bases than provided in sequence (" +
                                 std::to_string(base_count) + " > " + std::to_string(m_seq_len) +
                                 ").");
    }
    if (base_count < m_seq_len) {
        std::cerr << sequence << std::endl;
        throw std::runtime_error("Movement table indicates fewer bases than provided in sequence(" +
                                 std::to_string(base_count) + " < " + std::to_string(m_seq_len) +
                                 ").");
    }

    // Now we can go through each base and fill in where the 1s belong.
    std::vector<float> buffer(m_kmer_len * NUM_BASES);
    for (int seq_pos = -m_kmer_len + 1; seq_pos < m_seq_len; ++seq_pos) {
        // Fill buffer with the values corresponding to the kmer that begins with the current base.
        std::fill(buffer.begin(), buffer.end(), 0.0f);
        for (int kmer_pos = 0; kmer_pos < m_kmer_len; ++kmer_pos) {
            int this_base_pos = seq_pos + kmer_pos;
            int base_offset = -1;
            if (this_base_pos >= 0 && this_base_pos < m_seq_len)
                base_offset = m_base_mapping[sequence[this_base_pos]];
            if (base_offset == -1)
                continue;
            buffer[kmer_pos * NUM_BASES + base_offset] = 1.0f;
        }

        // Now we need to copy buffer into the encoded_data vector a number of times equal to the number of samples of
        // raw data corresponding to the kmer.
        int base_sample_pos = compute_sample_pos(seq_pos + m_bases_before);
        int next_base_sample_pos = compute_sample_pos(seq_pos + m_bases_before + 1);
        int num_repeats = next_base_sample_pos - base_sample_pos;

        // This is the position in the encoded data of the first sample corresponding to the kmer that begins with the
        // current base.
        int data_pos = base_sample_pos + m_padding * m_block_stride;
        if (data_pos + num_repeats > padded_signal_len) {
            throw std::runtime_error("Insufficient padding error.");
        }
        for (int i = 0; i < num_repeats; ++i, ++data_pos) {
            std::copy(buffer.begin(), buffer.end(),
                      m_encoded_data.data() + (data_pos * m_kmer_len * NUM_BASES));
        }
    }
}

RemoraEncoder::Context RemoraEncoder::get_context(size_t seq_pos) const {
    if (seq_pos >= size_t(m_seq_len)) {
        throw std::out_of_range("Sequence position out of range.");
    }
    Context context{};
    context.size = m_context_blocks * m_block_stride * m_kmer_len * NUM_BASES;
    int base_sample_pos =
            (compute_sample_pos(int(seq_pos)) + compute_sample_pos(int(seq_pos) + 1)) / 2;
    int samples_before = (m_context_blocks / 2) * m_block_stride;
    int first_sample = base_sample_pos - samples_before;
    if (first_sample >= 0) {
        context.first_sample = size_t(first_sample);
        context.lead_samples_needed = 0;
    } else {
        context.first_sample = 0;
        context.lead_samples_needed = size_t(-first_sample);
    }
    int last_sample = first_sample + m_context_blocks * m_block_stride;
    if (last_sample > m_signal_len) {
        context.num_samples = size_t(m_signal_len) - context.first_sample;
        context.tail_samples_needed = last_sample - m_signal_len;
    } else {
        context.num_samples = size_t(last_sample) - context.first_sample;
        context.tail_samples_needed = 0;
    }
    context.data = m_encoded_data.data() +
                   (m_padding * m_block_stride + first_sample) * m_kmer_len * NUM_BASES;
    return context;
}

int RemoraEncoder::compute_sample_pos(int base_pos) const {
    int base_offset = base_pos;
    if (base_offset < 0) {
        return m_block_stride * (base_offset);
    }
    if (base_offset >= m_seq_len) {
        return m_signal_len + m_block_stride * (base_offset - m_seq_len);
    }
    return m_sample_offsets[base_offset];
}
