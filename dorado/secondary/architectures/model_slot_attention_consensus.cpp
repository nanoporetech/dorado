#include "model_slot_attention_consensus.h"

#include "torch_utils/tensor_utils.h"

#include <spdlog/spdlog.h>

#include <cmath>
#include <random>
#include <stdexcept>
#include <tuple>

namespace dorado::secondary {

// Keeping the old quick_phase approach for now.
constexpr bool USE_BATCH_ADJACENCY_PHASE = true;

SlotAttentionImpl::SlotAttentionImpl(const int32_t num_slots,
                                     const int32_t dim,
                                     const int32_t iters,
                                     const float epsilon,
                                     const int32_t hidden_dim)
        : m_num_slots{num_slots},
          m_dim{dim},
          m_iters{iters},
          m_epsilon{epsilon},
          m_hidden_dim{hidden_dim},
          m_scale{static_cast<float>(std::pow(static_cast<double>(m_dim), -0.5))},
          m_slots_mu{register_parameter("slots_mu", torch::randn({1, 1, m_dim}))},
          m_slots_logsigma{register_parameter("slots_logsigma", torch::zeros({1, 1, m_dim}))} {
    torch::nn::init::xavier_uniform_(m_slots_logsigma);

    m_to_q = register_module("to_q", torch::nn::Linear(m_dim, m_dim));
    m_to_k = register_module("to_k", torch::nn::Linear(m_dim, m_dim));
    m_to_v = register_module("to_v", torch::nn::Linear(m_dim, m_dim));
    m_gru = register_module("gru", torch::nn::GRUCell(torch::nn::GRUCellOptions(m_dim, m_dim)));

    m_hidden_dim = std::max(m_dim, hidden_dim);

    m_mlp = register_module("mlp", torch::nn::Sequential{
                                           torch::nn::Linear(m_dim, m_hidden_dim),
                                           torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
                                           torch::nn::Linear(m_hidden_dim, m_dim),
                                   });

    m_norm_input = register_module("norm_input",
                                   torch::nn::LayerNorm(torch::nn::LayerNormOptions({m_dim})));
    m_norm_slots = register_module("norm_slots",
                                   torch::nn::LayerNorm(torch::nn::LayerNormOptions({m_dim})));
    m_norm_pre_ff = register_module("norm_pre_ff",
                                    torch::nn::LayerNorm(torch::nn::LayerNormOptions({m_dim})));

    // Initialize the noise.
    {
        // Seed the random generator
        std::mt19937 gen(m_seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        // Create and fill tensor manually
        at::Tensor noise_tensor = torch::empty({1, m_num_slots, dim}, torch::kFloat);
        for (int32_t i = 0; i < m_num_slots; ++i) {
            for (int32_t j = 0; j < m_dim; ++j) {
                noise_tensor[0][i][j] = dist(gen);
            }
        }

        // Register as a non-trainable parameter
        m_fixed_noise = register_parameter("fixed_noise", noise_tensor, /*requires_grad=*/false);
    }
}

std::pair<at::Tensor, at::Tensor> SlotAttentionImpl::forward(at::Tensor x,
                                                             at::Tensor mask,
                                                             const int32_t num_slots) {
    if (std::size(x.sizes()) != 3) {
        spdlog::warn(
                "Input tensor given to SlotAttention does not have exactly 3 dimensions. Shape: " +
                utils::tensor_shape_as_string(x) + ". Returning an unitialized tensor.");
        return {};
    }

    const auto device = x.device();
    const auto dtype = x.dtype();

    const int64_t b = x.size(0);
    const int64_t n = x.size(1);
    const int64_t d = x.size(2);
    const int32_t n_s = (num_slots > 0) ? num_slots : m_num_slots;

    mask = mask.unsqueeze(1).expand({-1, n_s, -1});

    // Bias.
    at::Tensor bias = torch::zeros_like(mask, torch::kFloat);
    bias.masked_fill_(mask, -std::numeric_limits<float>::infinity());

    // Computes: slots = mu + sigma * noise.
    at::Tensor slots = m_slots_mu.expand({b, n_s, -1}) +
                       m_slots_logsigma.exp().expand({b, n_s, -1}) *
                               m_fixed_noise.expand({b, -1, -1})
                                       .to(torch::TensorOptions().device(device).dtype(dtype));

    x = m_norm_input(x);

    const at::Tensor k = m_to_k(x);
    const at::Tensor v = m_to_v(x);

    at::Tensor attn;

    if (m_iters <= 0) {
        // Only allocate if needed.
        attn = torch::zeros({b, n_s, n}).to(torch::TensorOptions().device(device).dtype(dtype));
    }

    for (int32_t i = 0; i < m_iters; ++i) {
        const at::Tensor slots_prev = slots;

        slots = m_norm_slots(slots);
        at::Tensor q = m_to_q(slots);

        // s = n_slots, r = n_reads
        /// Original line from Medaka:
        ///     at::Tensor dots = torch::einsum("bsd,brd->bsr", {q, k}) * m_scale;
        /// Alternative, should be more efficient with the same result:
        at::Tensor dots = torch::bmm(q, k.transpose(1, 2)).mul_(m_scale).add_(bias);

        q = at::Tensor{};  // Clear memory.

        attn = torch::softmax(dots, 1).add_(m_epsilon);
        attn.masked_fill_(mask, 0);
        attn.div_(attn.nansum(-1, /*keepdim=*/true));

        dots = at::Tensor{};  // Clear memory.

        /// Original line from Medaka:
        ///     const at::Tensor updates = torch::einsum("bjd,bij->bid", {v, attn});
        /// Alternative using BMM (should be equivalent and more efficient):
        at::Tensor updates = torch::bmm(attn, v);

        slots = m_gru(updates.reshape({-1, d}), slots_prev.reshape({-1, d}));
        updates = at::Tensor{};  // Clear memory.

        slots = slots.reshape({b, -1, d});
        slots.add_(m_mlp->forward(m_norm_pre_ff(slots)));
    }

    return {slots, attn};
}

ModelSlotAttentionConsensus::ModelSlotAttentionConsensus(
        const MustConstructWithFactory& ctor_tag,
        const int32_t num_slots,
        const int32_t classes_per_slot,
        const int32_t read_embedding_size,
        const int32_t cnn_size,
        const std::vector<int32_t>& kernel_sizes,
        const std::string& pooler_type,
        const std::unordered_map<std::string, std::string>& pooler_args,
        const bool use_mapqc,
        const bool use_dwells,
        const bool use_haplotags,
        const int32_t bases_alphabet_size,
        const int32_t bases_embedding_size,
        const bool add_lstm,
        const bool use_reference,
        const FeatureColumnMap feature_column_map)
        : ModelTorchBase(ctor_tag),
          m_num_slots{num_slots},
          m_classes_per_slot{classes_per_slot},
          m_read_embedding_size{read_embedding_size},
          m_cnn_size{cnn_size},
          m_kernel_sizes{kernel_sizes},
          m_pooler_type{pooler_type},
          m_pooler_args{pooler_args},
          m_use_mapqc{use_mapqc},
          m_use_dwells{use_dwells},
          m_use_haplotags{use_haplotags},
          m_bases_alphabet_size{bases_alphabet_size},
          m_bases_embedding_size{bases_embedding_size},
          m_add_lstm{add_lstm},
          m_use_reference{use_reference},
          m_feature_column_map{feature_column_map},
          m_base_embedder{
                  torch::nn::EmbeddingOptions(m_bases_alphabet_size, m_bases_embedding_size)},
          m_haplotag_embedder{
                  torch::nn::EmbeddingOptions(MAX_HAPLOTAGS + 1, m_bases_embedding_size)},
          m_strand_embedder{torch::nn::EmbeddingOptions(3, m_bases_embedding_size)},
          m_read_level_conv{m_bases_embedding_size + (1 + m_use_dwells + m_use_mapqc),
                            m_read_embedding_size,
                            m_kernel_sizes,
                            std::vector<int32_t>(std::size(m_kernel_sizes), m_cnn_size),
                            true,
                            false},
          m_expansion_layer{m_cnn_size, m_read_embedding_size},
          m_slot_attention{m_num_slots, m_read_embedding_size, 3, 1e-8f, 128},
          m_slot_classifier{m_read_embedding_size, m_classes_per_slot},
          m_lstm{} {
    (void)m_use_reference;

    // Helper to get the feature column index.
    const auto get_feature_or_throw =
            [&feature_column_map](const FeatureColumns feature) -> int32_t {
        const auto it = feature_column_map.find(feature);
        if (it == std::cend(feature_column_map)) {
            throw std::runtime_error{"Cannot find the " + feature_column_to_string(feature) +
                                     " column in the feature_column_map!"};
        }
        return it->second;
    };

    // Mandatory feature columns.
    m_column_base = get_feature_or_throw(FeatureColumns::BASE);
    m_column_qual = get_feature_or_throw(FeatureColumns::QUAL);
    m_column_strand = get_feature_or_throw(FeatureColumns::STRAND);
    m_column_mapq = get_feature_or_throw(FeatureColumns::MAPQ);

    // Optional feature columns.
    m_column_dwell = use_dwells ? get_feature_or_throw(FeatureColumns::DWELL) : -1;
    m_column_haplotag = use_haplotags ? get_feature_or_throw(FeatureColumns::HAPLOTAG) : -1;

    if (m_add_lstm) {
        const int64_t lstm_size = m_num_slots * m_read_embedding_size;
        for (int32_t i = 0; i < 4; ++i) {
            m_lstm->push_back(ReversibleLSTM(lstm_size, lstm_size, true, !(i % 2)));
        }
        register_module("lstm", m_lstm);
    }

    register_module("base_embedder", m_base_embedder);
    register_module("haplotag_embedder", m_haplotag_embedder);
    register_module("strand_embedder", m_strand_embedder);
    register_module("read_level_conv", m_read_level_conv);
    register_module("expansion_layer", m_expansion_layer);
    register_module("slot_attention", m_slot_attention);
    register_module("slot_classifier", m_slot_classifier);
}

namespace {
at::Tensor batch_adjacency_phase(const at::Tensor& hap_probs_unphased,
                                 const torch::Tensor& basecalls,
                                 const int32_t lookback) {
    // Dimensions: (n_batch, n_pos, n_reads, n_feats, n_class).
    const int64_t n_pos = basecalls.size(1);

    basecalls.masked_fill_(basecalls == 0, -1);  // Remap padding bases.
    basecalls.masked_fill_(basecalls == 5, 0);   // Remap deletions to 0 as above.

    const at::Tensor adj_basecalls = basecalls.unfold(
            1, lookback + 1, 1);  // Shape (n_batch, n_pos-lookback, n_reads, lookback+1)
    at::Tensor hap_probs_phased = hap_probs_unphased.clone();
    at::Tensor hap_preds = hap_probs_phased.argmax(-1);  // (batch, pos, hap)
    const at::Tensor rolling_preds = hap_preds.unfold(
            1, lookback + 1,
            1);  // View into hap_preds, shape (batch, n_pos-lookback, lookback+1, n_haps)

    const at::Tensor swap_idx =
            torch::tensor({1, 0}, torch::TensorOptions().device(hap_probs_unphased.device()));

    using namespace torch::indexing;

    for (int64_t pos = lookback; pos < n_pos; ++pos) {
        const at::Tensor window_preds = rolling_preds.index(
                {torch::indexing::Slice(), pos - lookback});  // (batch, lookback+1, n_haps)
        const at::Tensor window_basecalls =
                adj_basecalls.index({torch::indexing::Slice(), pos - lookback});

        at::Tensor preds_flipped = window_preds.clone();
        preds_flipped.index_put_({torch::indexing::Ellipsis, -1},
                                 preds_flipped.index({torch::indexing::Slice(), swap_idx, -1}));

        const at::Tensor unflip_support = (window_preds.index({Slice(), 0})
                                                   .unsqueeze(-2)
                                                   .eq(window_basecalls)
                                                   .all(-1)
                                                   .sum(-1)) +
                                          (window_preds.index({Slice(), 1})
                                                   .unsqueeze(-2)
                                                   .eq(window_basecalls)
                                                   .all(-1)
                                                   .sum(-1));

        const at::Tensor flip_support = (preds_flipped.index({Slice(), 0})
                                                 .unsqueeze(-2)
                                                 .eq(window_basecalls)
                                                 .all(-1)
                                                 .sum(-1)) +
                                        (preds_flipped.index({Slice(), 1})
                                                 .unsqueeze(-2)
                                                 .eq(window_basecalls)
                                                 .all(-1)
                                                 .sum(-1));

        const at::Tensor flip_better = flip_support > unflip_support;

        if (flip_better.any().item<bool>()) {
            const at::Tensor idx = torch::nonzero(flip_better).squeeze(1);

            hap_probs_phased.index_put_(
                    {idx, pos}, hap_probs_phased.index({idx, pos}).index({Slice(), swap_idx}));

            hap_preds.index_put_({idx, pos},
                                 hap_preds.index({idx, pos}).index({Slice(), swap_idx}));
        }
    }

    return hap_probs_phased;
}
}  // namespace

at::Tensor ModelSlotAttentionConsensus::forward(torch::Tensor x) {
    auto [out, attn] = forward_impl(x);

    if constexpr (USE_BATCH_ADJACENCY_PHASE) {
        out = batch_adjacency_phase(out,
                                    x.index({torch::indexing::Ellipsis, m_column_base}).clone(), 4);
    } else {
        for (int64_t i = 0; i < x.size(0); ++i) {
            out[i] = quick_phase(out[i], attn[i], x[i]).first;
        }
    }

    return out;
}

double ModelSlotAttentionConsensus::estimate_batch_memory(
        const std::vector<int64_t>& batch_tensor_shape) const {
    if (std::size(batch_tensor_shape) != 4) {
        throw std::runtime_error{
                "Input tensor shape is of wrong dimension! Expected 4 sizes, got " +
                std::to_string(std::size(batch_tensor_shape))};
    }

    // Input tensor shape: [batch_size x num_positions x coverage x num_features];
    const int64_t batch_size = batch_tensor_shape[0];
    const int64_t num_positions = batch_tensor_shape[1];
    const int64_t coverage = batch_tensor_shape[2];

    // Limit the maximum batch size and maximum coverage to the bounds used for model estimation.
    constexpr int64_t MAX_BATCH_SIZE = 100;
    constexpr int64_t MAX_COVERAGE = 100;
    if ((batch_size > MAX_BATCH_SIZE) || (coverage > MAX_COVERAGE)) {
        return MEMORY_ESTIMATE_UPPER_CAP;
    }

    // IMPORTANT: The following equation was determined as part of the DOR-1350 effort.
    return (6.028445 * 1) + (0.000013 * num_positions) + (0.000020 * batch_size * num_positions) +
           (0.000003 * batch_size * num_positions * coverage) +
           (0.000027 * batch_size * std::pow(coverage, 2));
}

std::pair<at::Tensor, at::Tensor> ModelSlotAttentionConsensus::forward_impl(
        const torch::Tensor& in_x) {
    if (in_x.size(-1) != std::ssize(m_feature_column_map)) {
        throw std::runtime_error{
                "ModelSlotAttentionConsensus: x has the wrong number of feature columns! Feature "
                "column map is of size " +
                std::to_string(std::size(m_feature_column_map)) + " but got " +
                std::to_string(in_x.size(-1))};
    }
    if ((m_column_base < 0) || (m_column_qual < 0) || (m_column_strand < 0) ||
        (m_column_mapq < 0)) {
        throw std::runtime_error{
                "ModelSlotAttentionConsensus: One or more of the fixed feature indices is not "
                "valid! "
                "Indices: base = " +
                std::to_string(m_column_base) + ", qual = " + std::to_string(m_column_qual) +
                ", strand = " + std::to_string(m_column_strand) +
                ", mapq = " + std::to_string(m_column_mapq)};
    }
    if (m_use_dwells && (m_column_dwell < 0)) {
        throw std::runtime_error{
                "ModelSlotAttentionConsensus: The dwell column index is not valid! Got: " +
                std::to_string(m_column_dwell)};
    }
    if (m_use_haplotags && (m_column_haplotag < 0)) {
        throw std::runtime_error{
                "ModelSlotAttentionConsensus: The haplotag column index is not valid! Got: " +
                std::to_string(m_column_haplotag)};
    }

    // Bases embeddings.
    at::Tensor embeddings =
            m_base_embedder->forward(in_x.select(-1, m_column_base).to(torch::kLong));

    // Strand embeddings.
    embeddings.add_(
            m_strand_embedder->forward(in_x.select(-1, m_column_strand).to(torch::kLong) + 1));

    // Haplotag embeddings.
    if (m_use_haplotags) {
        embeddings.add_(
                m_haplotag_embedder->forward(in_x.select(-1, m_column_haplotag).to(torch::kLong)));
    }

    at::Tensor scaled_q_scores = (in_x.select(-1, m_column_qual) / 25).add_(-1).unsqueeze(-1);

    std::vector<at::Tensor> features{std::move(embeddings), std::move(scaled_q_scores)};

    if (m_use_mapqc) {
        at::Tensor scaled_mapqc = (in_x.select(-1, m_column_mapq) / 25).add(-1).unsqueeze(-1);
        features.emplace_back(std::move(scaled_mapqc));
    }

    if (m_use_dwells) {
        at::Tensor dwells = in_x.select(-1, m_column_dwell).unsqueeze(-1);
        features.emplace_back(std::move(dwells));
    }

    at::Tensor x = torch::cat(features, -1);

    features.clear();

    x = x.permute({0, 2, 3, 1});  // batch x reads x features x positions

    const int64_t b = x.size(0);
    const int64_t d = x.size(1);
    // const int64_t f = x.size(2);
    const int64_t p = x.size(3);

    x = x.flatten(0, 1);
    x = m_read_level_conv(x);
    x = x.view({b, d, -1, p});
    x = x.permute({0, 3, 1, 2}).flatten(0, 1);
    x = m_expansion_layer(x);

    at::Tensor empty_position_mask = in_x.select(-1, m_column_base).eq(0).flatten(0, 1);

    at::Tensor attn;
    std::tie(x, attn) =
            m_slot_attention->forward(std::move(x), std::move(empty_position_mask), m_num_slots);

    x = x.view({b, p, m_num_slots, -1});
    attn = attn.view({b, p, m_num_slots, d});

    if (m_add_lstm) {
        const at::Tensor lstm_input = x.flatten(-2);
        torch::IValue delta_ival = m_lstm->forward(lstm_input);
        at::Tensor delta;
        if (delta_ival.isTuple()) {
            delta = delta_ival.toTuple()->elements()[0].toTensor();
        } else {
            delta = delta_ival.toTensor();
        }
        delta = delta.view({delta.size(0), delta.size(1), m_num_slots, -1});
        x += delta;
    }

    x = m_slot_classifier(x);

    if (m_normalise_before_phasing) {
        x = torch::softmax(x, -1);
    }

    return {x, attn};
}

std::pair<at::Tensor, at::Tensor> ModelSlotAttentionConsensus::quick_phase(
        at::Tensor hap_probs_unphased,
        const at::Tensor& attn,
        const at::Tensor& features) const {
    if ((attn.size(0) != hap_probs_unphased.size(0)) ||
        (attn.size(1) != hap_probs_unphased.size(1))) {
        throw std::runtime_error{
                "Attention and logits have different number of positions or haplotypes. attn.shape "
                "= {" +
                utils::tensor_shape_as_string(attn) + "}, hap_probs_unphased.shape = {" +
                utils::tensor_shape_as_string(hap_probs_unphased) + "}"};
    }

    const auto device = attn.device();

    const int64_t n_pos = attn.size(0);
    const int64_t n_haps = attn.size(1);
    const int64_t n_reads = attn.size(2);
    const int64_t n_class = hap_probs_unphased.size(2);

    // Consider only reads that span the entire window.
    const at::Tensor span_reads_mask = (features.select(-1, 0) == 0).sum(0) == 0;
    const int64_t n_span_reads = span_reads_mask.sum().item<int64_t>();
    if (n_span_reads < 2) {
        return {std::move(hap_probs_unphased), torch::zeros(n_reads)};
    }

    // Identify heterozygous positions.
    const at::Tensor hap_preds = hap_probs_unphased.argmax(-1);
    if (hap_preds.size(-1) != 2) {
        throw std::runtime_error("Expected 2 haplotypes but found " +
                                 std::to_string(hap_preds.size(-1)) + "! Not implemented yet.");
    }
    const at::Tensor het_pos_mask = (hap_preds.index({torch::indexing::Slice(), 0}) !=
                                     hap_preds.index({torch::indexing::Slice(), 1}));
    const int64_t n_het_pos = het_pos_mask.sum().item<int64_t>();
    if (n_het_pos <= 1) {
        // Nothing to phase.
        return {std::move(hap_probs_unphased), torch::zeros(n_reads)};
    }

    // -------------------------------------
    // 1. Cluster reads to assign haplotags.
    // -------------------------------------
    // Use only heterozygous positions. The attention tensor for these positions
    // is reshaped so that each read is represented by a vector over the het positions.
    // Shape after indexing: (n_het_pos, n_haps, n_reads) → reshape to (n_reads, n_het_pos*n_haps)
    // Here, we simply collapse the heterozygous positions; other schemes are possible.
    at::Tensor attn_span_reads =
            attn.index({at::indexing::Slice(), at::indexing::Slice(), span_reads_mask});
    at::Tensor het_attn_span_reads = attn_span_reads.index({het_pos_mask})
                                             .reshape({-1, n_span_reads})
                                             .transpose(0, 1);  // Shape: (n_reads, ?)
    // const at::Tensor span_read_haps = kmeans_cluster(het_attn_span_reads, 2, 10).to(torch::TensorOptions().device(device).dtype(dtype));;
    at::Tensor span_read_haps = kmeans_cluster(het_attn_span_reads, 2, 10);
    at::Tensor read_haps =
            torch::zeros(n_reads).to(torch::TensorOptions().device(device).dtype(torch::kLong));
    read_haps.index_put_({span_reads_mask}, span_read_haps + 1);

    // -------------------------------------
    // 2. Compute permutation scores in a vectorized way.
    // -------------------------------------
    // Build all possible permutations of haplotype indices.
    at::Tensor perms =
            torch::tensor({{0, 1}, {1, 0}}, torch::TensorOptions().device(attn.device()));
    const int64_t num_perm = perms.size(0);

    // Create an indicator (one-hot) for read haplotags.
    // Shape: (n_reads, n_haps) --> transpose to (n_haps, n_reads)
    at::Tensor indicators = torch::nn::functional::one_hot(span_read_haps, n_haps)
                                    .to(torch::kFloat)
                                    .transpose(0, 1);  // (n_haps, n_reads)
    span_read_haps = at::Tensor{};                     // Clear memory.

    // Expand attn and perms so that we can gather all permuted versions at once.
    // Our goal is to compute, for each position i and each permutation p, the score:
    //   score[i, p] = sum_{j=0}^{n_haps-1} dot( attn[i, perms[p, j], :], indicators[j, :] )
    //
    // First, add a permutation dimension to attn and expand:
    // attn: (n_pos, n_haps, n_reads) → (n_pos, 1, n_haps, n_reads)
    at::Tensor attn_exp =
            attn_span_reads.unsqueeze(1).expand({n_pos, num_perm, n_haps, n_span_reads});
    attn_span_reads = at::Tensor{};  // Clear memory.
    // Expand perms: (num_perm, n_haps) → (1, num_perm, n_haps, 1)
    at::Tensor perms_exp =
            perms.unsqueeze(0).unsqueeze(-1).expand({n_pos, num_perm, n_haps, n_span_reads});
    // Gather the permuted attention scores along the haplotype dimension.
    // After gathering, attn_perm[i, p, j, :] = attn[i, perms[p, j], :]
    at::Tensor attn_perm =
            torch::gather(attn_exp, 2, perms_exp);  // Shape: (n_pos, num_perm, n_haps, n_reads)
    perms_exp = at::Tensor{};                       // Clear memory.

    // Multiply by the appropriate indicator for each haplotag slot.
    // indicators: (n_haps, n_reads) → (1, 1, n_haps, n_reads)
    at::Tensor attn_weighted = attn_perm * indicators.unsqueeze(0).unsqueeze(0);
    attn_perm = at::Tensor{};   // Clear memory.
    attn_exp = at::Tensor{};    // Clear memory.
    indicators = at::Tensor{};  // Clear memory.
    // Sum over haplotag and read dimensions to get a score for each position and permutation.
    // permutation_scores: (n_pos, num_perm)
    at::Tensor permutation_scores = attn_weighted.sum(3).sum(2);
    attn_weighted = at::Tensor{};  // Clear memory.

    // -------------------------------------
    // 3. Select the best permutation per position.
    // -------------------------------------
    // For each position, choose the permutation index that gives the highest score.
    at::Tensor best_perm_idx = permutation_scores.argmax(1);  // Shape: (n_pos,)
    permutation_scores = at::Tensor{};                        // Clear memory.
    // Look up the actual permutation indices. best_perms: (n_pos, n_haps)
    at::Tensor best_perms = perms.index_select(0, best_perm_idx);
    perms = at::Tensor{};          // Clear memory.
    best_perm_idx = at::Tensor{};  // Clear memory.

    // -------------------------------------
    // 4. Reorder the haplotype probabilities.
    // -------------------------------------
    // hap_probs_unphased: (n_pos, n_haps, n_class)
    // We want, for each position i, to select hap_probs_unphased[i, best_perms[i]]
    // Use torch.gather along the haplotype (dim=1).
    at::Tensor best_perms_exp = best_perms.unsqueeze(-1).expand({n_pos, n_haps, n_class});
    at::Tensor phased_hap_probs = torch::gather(hap_probs_unphased, 1, best_perms_exp);
    best_perms = at::Tensor{};      // Clear memory.
    best_perms_exp = at::Tensor{};  // Clear memory.

    return {std::move(phased_hap_probs), std::move(read_haps)};
}

namespace {
/// @brief Calculate pairwise euclidean distance matrix between two vector lists.
/// @param tensor_x tensor of shape (n, d) where n is the number of vectors
/// @param tensor_y tensor of shape (m, d) where m is the number of vectors
/// @return tensor of shape (n, m) where each element is the distance between the corresponding vectors in x and y.
at::Tensor distance_matrix(const at::Tensor& tensor_x, const at::Tensor& tensor_y) {
    const int64_t n = tensor_x.size(0);
    const int64_t m = tensor_y.size(0);

    const at::Tensor x = tensor_x.unsqueeze(1).expand({n, m, -1});
    const at::Tensor y = tensor_y.unsqueeze(0).expand({n, m, -1});

    return (x - y).norm(2, -1);
}
}  // namespace

at::Tensor ModelSlotAttentionConsensus::kmeans_cluster(const at::Tensor& x,
                                                       const int32_t k,
                                                       const int32_t n_iters) const {
    const at::Tensor sorted_indices = torch::argsort(x.select(1, 0));
    at::Tensor cluster_points = x.index_select(0, sorted_indices.slice(0, 0, k));

    for (int32_t i = 0; i < n_iters; ++i) {
        const at::Tensor labels = distance_matrix(x, cluster_points).argmin(1);
        for (int32_t lab = 0; lab < k; ++lab) {
            const at::Tensor select = (labels == lab);
            cluster_points.index_put_({lab}, x.index_select(0, select.nonzero().squeeze()).mean(0));
        }
    }

    at::Tensor labels = distance_matrix(x, cluster_points).argmin(1);

    return labels;
}

}  // namespace dorado::secondary
