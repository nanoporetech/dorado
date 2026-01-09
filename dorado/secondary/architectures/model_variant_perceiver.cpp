#include "model_variant_perceiver.h"

#include "torch_utils/gpu_profiling.h"
#include "torch_utils/tensor_utils.h"

#include <spdlog/spdlog.h>

#include <cmath>
#include <random>
#include <stdexcept>
#include <tuple>

// #define DEBUG_VARIANT_PERCEIVER_DATA_TYPES

#ifdef NDEBUG
#define LOG_TRACE(...)
#else
#define LOG_TRACE(...) spdlog::trace(__VA_ARGS__)
#endif

#ifndef DEBUG_VARIANT_PERCEIVER_DATA_TYPES
#define LOG_TRACE_DTYPE(...)
#else
#define LOG_TRACE_DTYPE(...) spdlog::trace(__VA_ARGS__)
#endif

namespace dorado::secondary {

SwiGLUImpl::SwiGLUImpl(const int32_t in_features, const int32_t hidden_features, const bool bias) {
    m_fc1 = register_module(
            "fc1", torch::nn::Linear(
                           torch::nn::LinearOptions(in_features, 2 * hidden_features).bias(bias)));
    m_fc2 = register_module(
            "fc2",
            torch::nn::Linear(torch::nn::LinearOptions(hidden_features, in_features).bias(bias)));
};

at::Tensor SwiGLUImpl::forward(const at::Tensor& x) {
    utils::ScopedProfileRange spr1("SwiGLUImpl::forward", 4);
    at::Tensor t = m_fc1(x);
    const auto chunks = t.chunk(2, -1);
    const auto& y = chunks[0];
    const auto& gate = chunks[1];
    t = torch::nn::functional::silu(gate).mul_(y);
    LOG_TRACE_DTYPE("[SwiGLUImpl] x.dtype() = {}, t.dtype() = {}", torch::toString(x.scalar_type()),
                    torch::toString(t.scalar_type()));
    return m_fc2(t);
}

RotaryEmbeddingImpl::RotaryEmbeddingImpl(const int64_t dim,
                                         const float theta,
                                         const at::TensorOptions& options)
        : m_dim{dim}, m_theta{theta} {
    if (dim <= 0) {
        throw std::runtime_error{"Value of dim not valid for RotaryEmbedding. dim = " +
                                 std::to_string(dim) + ", should be > 0."};
    }

    m_inv_freq =
            torch::pow(m_theta, torch::arange(0, m_dim, 2, options) / static_cast<float>(m_dim))
                    .reciprocal()
                    .detach();

    // NOTE: There is no `persistent` option in Libtorch unlike Pytorch:
    //      register_buffer("inv_freq", m_inv_freq, /*persistent=*/false);
    // Registering a buffer means it will be visible in the named_buffers and might cause
    // problems when loading weights.
    // But if the buffer is not registered, then it will not get moved to the target device like
    // other parameters, and this will crash execution.
    // Workaround: there is now a manually added `add_nonpersistent_buffer()` function in the
    // ModelTorchBase, and the top-level model logs this buffer, so that it can be checked later.
    register_buffer("inv_freq", m_inv_freq);
}

std::pair<at::Tensor, at::Tensor> RotaryEmbeddingImpl::forward(at::Tensor q, at::Tensor k) {
    utils::ScopedProfileRange spr1("RotaryEmbeddingImpl::forward", 4);

    if (std::size(q.sizes()) != std::size(k.sizes())) {
        throw std::runtime_error{"Q and K tensors mismatch in dimensions! q.shape = " +
                                 utils::tensor_shape_as_string(q) +
                                 ", k.shape = " + utils::tensor_shape_as_string(k)};
    }
    if (std::size(q.sizes()) != 5) {
        throw std::runtime_error{"Q and K tensors should be 5D. Given: q.shape = " +
                                 utils::tensor_shape_as_string(q)};
    }
    if (!q.defined()) {
        throw std::runtime_error{"Cannot run RotaryEmbedding::forward on an undefined q tensor."};
    }
    if (!k.defined()) {
        throw std::runtime_error{"Cannot run RotaryEmbedding::forward on an undefined k tensor."};
    }

    LOG_TRACE_DTYPE("[RotaryEmbeddingImpl] Input: q.dtype() = {}, k.dtype() = {}",
                    torch::toString(q.scalar_type()), torch::toString(k.scalar_type()));

    // Dimensions: N, T, C, H, D = batch_size, num_positions, num_sequences, num_heads, head_dim
    const int64_t T = q.size(1);

    // TODO: Cache the freqs computation similar to TxModules.
    const at::Tensor pos = torch::arange(T, q.options());
    const at::Tensor freqs = at::outer(
            pos, m_inv_freq);  // Equivalent to: torch::einsum("i,j->ij", {pos, m_inv_freq});
    const at::Tensor emb = torch::cat({freqs, freqs}, /*dim=*/-1);

    // emb: [L, D]
    const at::Tensor cos_vals = torch::cos(emb)         // [T, D]
                                        .unsqueeze(0)   // [1, T, D]
                                        .unsqueeze(2)   // [1, T, 1, D]
                                        .unsqueeze(3);  // [1, T, 1, 1, D]

    const at::Tensor sin_vals = torch::sin(emb).unsqueeze(0).unsqueeze(2).unsqueeze(3);

    q = q * cos_vals + rotate_half(q) * sin_vals;
    k = k * cos_vals + rotate_half(k) * sin_vals;

    LOG_TRACE_DTYPE("[RotaryEmbeddingImpl] Output: q.dtype() = {}, k.dtype() = {}",
                    torch::toString(q.scalar_type()), torch::toString(k.scalar_type()));

    return {std::move(q), std::move(k)};
}

at::Tensor RotaryEmbeddingImpl::rotate_half(const at::Tensor& x) const {
    std::vector<at::Tensor> chunks = x.chunk(2, /*dim=*/-1);
    return torch::cat({-chunks[1], chunks[0]}, /*dim=*/-1);
}

MultiSequenceCrossAttentionBlockImpl::MultiSequenceCrossAttentionBlockImpl(
        const int64_t dim,
        const int64_t /*ploidy*/ /*=2*/,
        const int64_t /*n_pos*/ /*=100*/,
        const int64_t num_heads /*=4*/,
        const int64_t max_depth /*=100*/,
        const float /* dropout =0.1*/,
        const bool qkv_bias, /*=false*/
        const std::optional<int64_t>& attn_window)
        : m_num_heads{num_heads}, m_attn_window{attn_window} {
    if (num_heads <= 0) {
        throw std::runtime_error{"Number of heads should be > 0, given: " +
                                 std::to_string(num_heads)};
    }

    m_head_dim = dim / num_heads;

    m_kv_proj = register_module(
            "kv_proj", torch::nn::Linear(torch::nn::LinearOptions(dim, dim * 2).bias(qkv_bias)));
    m_q_proj = register_module(
            "q_proj", torch::nn::Linear(torch::nn::LinearOptions(dim, dim).bias(qkv_bias)));
    m_read_embeddings = register_module("read_embeddings", torch::nn::Embedding(max_depth, dim));
    m_positional_embeddings = register_module(
            "positional_embeddings", RotaryEmbedding(m_head_dim, 10000.0f, at::TensorOptions{}));
    m_out_proj = register_module("out_proj", SwiGLU(dim, dim, false));
    m_norm1 = register_module("norm1", nn::RMSNorm(dim));
    m_norm2 = register_module("norm2", nn::RMSNorm(dim));
}

at::Tensor MultiSequenceCrossAttentionBlockImpl::local_attention_mask(
        const int64_t T,
        const int64_t num_q_seqs,
        const int64_t num_kv_seqs,
        const int64_t attn_window) const {
    TORCH_CHECK(T > 0, "T must be > 0 (got ", T, ")");
    TORCH_CHECK(num_q_seqs >= 0, "num_q_seqs must be >= 0 (got ", num_q_seqs, ")");
    TORCH_CHECK(num_kv_seqs >= 0, "num_kv_seqs must be >= 0 (got ", num_kv_seqs, ")");
    TORCH_CHECK(attn_window >= 0, "attn_window must be >= 0 (got ", attn_window, ")");

    const int64_t Q_LEN = T * num_q_seqs;
    const int64_t KV_LEN = T * num_kv_seqs;

    const auto device = this->parameters()[0].device();
    const auto opts = torch::TensorOptions().dtype(torch::kInt64).device(device);

    // q_idx: [Q_LEN], k_idx: [KV_LEN]
    const at::Tensor q_idx = torch::arange(Q_LEN, opts);
    const at::Tensor k_idx = torch::arange(KV_LEN, opts);

    // q_pos = q_idx % T, k_pos = k_idx % T
    const at::Tensor q_pos = torch::remainder(q_idx, T);  // [Q_LEN]
    const at::Tensor k_pos = torch::remainder(k_idx, T);  // [KV_LEN]

    // Broadcast difference to [Q_LEN, KV_LEN]
    const at::Tensor diff = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).abs();

    // mask: bool [Q_LEN, KV_LEN]
    at::Tensor mask = diff.le(attn_window);

    return mask;
}

at::Tensor MultiSequenceCrossAttentionBlockImpl::attn_fn(const at::Tensor& q,
                                                         const at::Tensor& k,
                                                         const at::Tensor& v) const {
    /**
     * q shape: N, T, N_Q, H, D (batch_size, num_positions, num_query_seqs, num_heads, head_dim)
     * k shape: N, T, N_KV, H, D (batch_size, num_positions, num_kv_seqs, num_heads, head_dim)
     */

    const int64_t N = q.size(0);
    const int64_t T = q.size(1);
    const int64_t N_Q = q.size(2);
    const int64_t H = q.size(3);
    const int64_t D = q.size(4);
    const int64_t N_KV = k.size(2);

    // Flatten positions and sequences, keep batch and head:
    //      q -> (N, H, Lq, D), k/v -> (N, H, Lkv, D)
    const at::Tensor q2 = q.permute({0, 3, 2, 1, 4}).contiguous().view({N, H, N_Q * T, D});
    const at::Tensor k2 = k.permute({0, 3, 2, 1, 4}).contiguous().view({N, H, N_KV * T, D});
    const at::Tensor v2 = v.permute({0, 3, 2, 1, 4}).contiguous().view({N, H, N_KV * T, D});

    std::optional<at::Tensor> mask{std::nullopt};

    // Compute mask if needed.
    if (m_attn_window) {
        at::Tensor new_mask = local_attention_mask(T, N_Q, N_KV, *m_attn_window);

        // Reshape the mask. Expand broadcasts from [1, 1, ...] to [N, H, ...]. It doesn't copy the data,
        // it just produces a view of those rows.
        new_mask = new_mask.view({1, 1, N_Q * T, N_KV * T})
                           .expand({N, H, N_Q * T, N_KV * T});  // [1, 1, Lq, Lkv]
        mask = {std::move(new_mask)};
    }

    LOG_TRACE_DTYPE("[attn] q.dtype() = {}, k.dtype() = {}, v.dtype() = {}",
                    torch::toString(q.scalar_type()), torch::toString(k.scalar_type()),
                    torch::toString(v.scalar_type()));
    LOG_TRACE_DTYPE("[attn] q2.dtype() = {}, k2.dtype() = {}, v2.dtype() = {}",
                    torch::toString(q2.scalar_type()), torch::toString(k2.scalar_type()),
                    torch::toString(v2.scalar_type()));

    // Convert to half for more efficient attention calculation.
    const auto qh = q2.to(at::kBFloat16);
    const auto kh = k2.to(at::kBFloat16);
    const auto vh = v2.to(at::kBFloat16);

    at::Tensor attn = at::scaled_dot_product_attention(qh, kh, vh, mask);

    attn = attn.to(at::kFloat);

    // Reshape back to (N, T, N_Q, H*D).
    attn = attn.view({N, H, N_Q, T, D}).permute({0, 3, 2, 1, 4}).contiguous();  // (N, T, N_Q, H, D)

    at::Tensor out = attn.view({N, T, N_Q, H * D});  // (N, T, N_Q, dim)

    return out;
}

at::Tensor MultiSequenceCrossAttentionBlockImpl::forward(at::Tensor x,
                                                         const at::Tensor& cross_attn_seqs) {
    utils::ScopedProfileRange spr1("MultiSequenceCrossAttentionBlockImpl::forward", 3);

    const int64_t N = x.size(0);
    const int64_t T = x.size(1);
    const int64_t N_Q = x.size(2);
    const int64_t N_KV = cross_attn_seqs.size(2);

    // Get the Q tensor.
    const at::Tensor q = m_q_proj(x).view({N, T, N_Q, m_num_heads, m_head_dim});

    // Get the K, V tensors.
    const at::Tensor kv = m_kv_proj(cross_attn_seqs).view({N, T, N_KV, 2, m_num_heads, m_head_dim});
    std::vector<torch::Tensor> kv_unbound = kv.unbind(/*dim=*/3);
    if (std::ssize(kv_unbound) != 2) {
        throw std::runtime_error{"Wrong size of the unbound tensors! kv_unbound.size = " +
                                 std::to_string(std::size(kv_unbound)) + ", expected = 2"};
    }
    const auto& k = kv_unbound[0];
    const auto& v = kv_unbound[1];

    const auto [q_rot, k_rot] = m_positional_embeddings(q, k);

    LOG_TRACE_DTYPE(
            "[MultiSequenceCrossAttentionBlockImpl] x.dtype() = {}, q.dtype() = {}, k.dtype() = "
            "{}, v.dtype() = {}",
            torch::toString(x.scalar_type()), torch::toString(q.scalar_type()),
            torch::toString(k.scalar_type()), torch::toString(v.scalar_type()));

    const at::Tensor attn_out = attn_fn(q_rot, k_rot, v);

    x = m_norm1(x + attn_out);
    x = m_norm2(m_out_proj(x) + x);

    LOG_TRACE_DTYPE("[MultiSequenceCrossAttentionBlockImpl] Output: x.dtype() = {}",
                    torch::toString(x.scalar_type()));

    return x;
}

SelfAttentionBlockImpl::SelfAttentionBlockImpl(const int64_t dim,
                                               const int64_t num_heads,
                                               const float dropout,
                                               const std::optional<int64_t>& attn_window) {
    m_self_attention = register_module(
            "self_attention", MultiSequenceCrossAttentionBlock(dim, 2, 100, num_heads, 100, dropout,
                                                               false, attn_window));
    m_norm = register_module("norm", nn::RMSNorm(dim));
}

at::Tensor SelfAttentionBlockImpl::forward(const at::Tensor& x) {
    utils::ScopedProfileRange spr1("SelfAttentionBlockImpl::forward", 3);
    at::Tensor ret = m_norm(x + m_self_attention(x, x));
    LOG_TRACE_DTYPE("[SelfAttentionBlockImpl] x.dtype() = {}, ret.dtype() = {}",
                    torch::toString(x.scalar_type()), torch::toString(ret.scalar_type()));
    return ret;
}

MessagePassingBlockImpl::MessagePassingBlockImpl(const int64_t dim,
                                                 const int64_t num_heads,
                                                 const float dropout,
                                                 const bool update_read_embeddings,
                                                 const bool cross_attend_read_embeddings,
                                                 const std::optional<int64_t>& attn_window)
        : m_update_read_embeddings{update_read_embeddings},
          m_cross_attend_read_embeddings{cross_attend_read_embeddings} {
    if (m_cross_attend_read_embeddings) {
        // Use the attention window in the cross attention.
        m_reads_to_haplotypes =
                register_module("reads_to_haplotypes",
                                MultiSequenceCrossAttentionBlock(dim, 2, 100, num_heads, 100,
                                                                 dropout, false, attn_window));
    }

    // No attention window in the self attention.
    m_haplotype_self_attention = register_module(
            "haplotype_self_attention", SelfAttentionBlock(dim, num_heads, dropout, std::nullopt));

    if (m_update_read_embeddings) {
        // Use the attention window in the cross attention.
        m_haplotypes_to_reads =
                register_module("haplotypes_to_reads",
                                MultiSequenceCrossAttentionBlock(dim, 2, 100, num_heads, 100,
                                                                 dropout, false, attn_window));
    }

    m_norm_1 = register_module("norm1", nn::RMSNorm(dim));
    m_norm_2 = register_module("norm2", nn::RMSNorm(dim));
}

std::pair<at::Tensor, at::Tensor> MessagePassingBlockImpl::forward(at::Tensor read_seqs,
                                                                   at::Tensor hap_seqs) {
    utils::ScopedProfileRange spr1("MessagePassingBlockImpl::forward", 2);

    LOG_TRACE_DTYPE(
            "[MessagePassingBlockImpl] Input: hap_seqs.dtype() = {}, read_seqs.dtype() = {}",
            torch::toString(hap_seqs.scalar_type()), torch::toString(read_seqs.scalar_type()));

    if (m_cross_attend_read_embeddings) {
        hap_seqs = m_reads_to_haplotypes(hap_seqs, read_seqs);

        LOG_TRACE_DTYPE(
                "[MessagePassingBlockImpl] Cross-attention (hap_seqs): hap_seqs.dtype() = {}, "
                "read_seqs.dtype() = {}",
                torch::toString(hap_seqs.scalar_type()), torch::toString(read_seqs.scalar_type()));
    }

    hap_seqs = m_haplotype_self_attention(hap_seqs);

    LOG_TRACE_DTYPE(
            "[MessagePassingBlockImpl] Self-attention (hap_seqs): hap_seqs.dtype() = {}, "
            "read_seqs.dtype() = {}",
            torch::toString(hap_seqs.scalar_type()), torch::toString(read_seqs.scalar_type()));

    if (m_update_read_embeddings) {
        read_seqs = m_haplotypes_to_reads(read_seqs, hap_seqs);

        LOG_TRACE_DTYPE(
                "[MessagePassingBlockImpl] Update embeddings (read_seqs): hap_seqs.dtype() = {}, "
                "read_seqs.dtype() = {}",
                torch::toString(hap_seqs.scalar_type()), torch::toString(read_seqs.scalar_type()));
    }

    return {read_seqs, hap_seqs};
}

ModelVariantPerceiver::ModelVariantPerceiver(const MustConstructWithFactory& ctor_tag,
                                             const int32_t ploidy,
                                             const int32_t num_classes,
                                             const int32_t read_embedding_size,
                                             const int32_t cnn_size,
                                             const std::vector<int32_t>& kernel_sizes,
                                             const int32_t dimension,
                                             const int32_t num_blocks,
                                             const int32_t num_heads,
                                             const bool use_mapqc,
                                             const bool use_dwells,
                                             const bool use_haplotags,
                                             const bool use_snp_qv,
                                             const int32_t bases_alphabet_size,
                                             const int32_t bases_embedding_size,
                                             // bool time_steps,
                                             const bool use_decoder_lstm,
                                             const bool update_read_embeddings,
                                             const FeatureColumnMap& feature_column_map)
        : ModelTorchBase(ctor_tag),
          m_ploidy{ploidy},
          m_num_classes{num_classes},
          m_read_embedding_size{read_embedding_size},
          m_cnn_size{cnn_size},
          m_kernel_sizes{kernel_sizes},
          m_dimension{dimension},
          m_num_blocks{num_blocks},
          m_num_heads{num_heads},
          m_use_mapqc{use_mapqc},
          m_use_dwells{use_dwells},
          m_use_haplotags{use_haplotags},
          m_use_snp_qv{use_snp_qv},
          m_bases_alphabet_size{bases_alphabet_size},
          m_bases_embedding_size{bases_embedding_size},
          m_use_decoder_lstm{use_decoder_lstm},
          m_update_read_embeddings{update_read_embeddings},
          m_feature_column_map{feature_column_map},
          m_base_embedder{
                  torch::nn::EmbeddingOptions(m_bases_alphabet_size, m_bases_embedding_size)},
          m_haplotag_embedder{
                  torch::nn::EmbeddingOptions(MAX_HAPLOTAGS + 1, m_bases_embedding_size)},
          m_strand_embedder{torch::nn::EmbeddingOptions(3, m_bases_embedding_size)},
          m_read_level_conv{
                  m_bases_embedding_size + (1 + m_use_dwells + m_use_mapqc + m_use_snp_qv),
                  m_read_embedding_size,
                  m_kernel_sizes,
                  std::vector<int32_t>(std::size(m_kernel_sizes), m_cnn_size),
                  /*use_batch_norm = */ true,
                  /*add_expansion_layer = */ false},
          m_expansion_layer{m_cnn_size, m_dimension},
          m_latent_init{torch::randn(m_dimension)},
          m_blocks{},
          m_decoder_identity{},
          m_output{m_dimension, m_num_classes * m_ploidy} {
    for (int32_t i = 0; i < m_num_blocks; ++i) {
        constexpr float CURR_DROPOUT = 0.1f;
        constexpr bool CURR_CROSS_ATTEND = true;
        const bool curr_update = (i < (m_num_blocks - 1)) ? m_update_read_embeddings : false;
        const std::optional<int64_t> curr_attn_window = std::nullopt;
        // blocks.emplace_back(
        MessagePassingBlock block(m_dimension, m_num_heads, CURR_DROPOUT, curr_update,
                                  CURR_CROSS_ATTEND, curr_attn_window);
        m_blocks->push_back(block);

        // Manually store the names of the non-persistent buffers because Libtorch doesn't have this feature (unlike Pytorch).
        // This will be cross-referenced during model loading.
        this->add_nonpersistent_buffer("blocks." + std::to_string(i) +
                                       ".reads_to_haplotypes.positional_embeddings.inv_freq");
        this->add_nonpersistent_buffer(
                "blocks." + std::to_string(i) +
                ".haplotype_self_attention.self_attention.positional_embeddings.inv_freq");
        if (curr_update) {
            this->add_nonpersistent_buffer("blocks." + std::to_string(i) +
                                           ".haplotypes_to_reads.positional_embeddings.inv_freq");
        }
    }

    if (use_decoder_lstm) {
        m_decoder_lstm =
                torch::nn::LSTM(torch::nn::LSTMOptions(/*input_size=*/m_dimension,
                                                       /*hidden_size=*/m_dimension)
                                        .num_layers(1)
                                        .batch_first(true)     // matches batch_first=True
                                        .bidirectional(false)  // matches bidirectional=False
                );
        register_module("decoder", m_decoder_lstm);
    } else {
        register_module("decoder", m_decoder_identity);
    }

    register_module("base_embedder", m_base_embedder);
    register_module("haplotag_embedder", m_haplotag_embedder);
    register_module("strand_embedder", m_strand_embedder);
    register_module("read_level_conv", m_read_level_conv);
    register_module("expansion_layer", m_expansion_layer);
    register_module("blocks", m_blocks);
    register_module("output", m_output);
    register_parameter("latent_init", m_latent_init);

    // Mandatory feature columns.
    m_column_base = get_feature_column_or_throw(feature_column_map, FeatureColumns::BASE);
    m_column_qual = get_feature_column_or_throw(feature_column_map, FeatureColumns::QUAL);
    m_column_strand = get_feature_column_or_throw(feature_column_map, FeatureColumns::STRAND);
    m_column_mapq = get_feature_column_or_throw(feature_column_map, FeatureColumns::MAPQ);

    // Optional feature columns.
    m_column_dwell =
            use_dwells ? get_feature_column_or_throw(feature_column_map, FeatureColumns::DWELL)
                       : -1;
    m_column_haplotag = use_haplotags ? get_feature_column_or_throw(feature_column_map,
                                                                    FeatureColumns::HAPLOTAG)
                                      : -1;
    m_column_snp_qv =
            use_snp_qv ? get_feature_column_or_throw(feature_column_map, FeatureColumns::SNP_QV)
                       : -1;
}

at::Tensor ModelVariantPerceiver::forward(at::Tensor x) { return forward_impl(x); }

double ModelVariantPerceiver::estimate_batch_memory(
        const std::vector<int64_t>& batch_tensor_shape) const {
    // TODO: These memory estimates are from the ModelSlotAttentionConsensus which seems to
    // be more memory hungry, thus making these pessimistic.
    // Reanalyze the memory consumption to update the equation.

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

    double ret = (6.028445 * 1) + (0.000013 * num_positions) +
                 (0.000020 * batch_size * num_positions) +
                 (0.000003 * batch_size * num_positions * coverage) +
                 (0.000027 * batch_size * std::pow(coverage, 2));

    return ret;
}

void ModelVariantPerceiver::validate_feature_tensor(const at::Tensor& x) const {
    if (x.size(-1) != std::ssize(m_feature_column_map)) {
        throw std::runtime_error{
                "ModelVariantPerceiver: x has the wrong number of feature columns! Feature "
                "column map is of size " +
                std::to_string(std::size(m_feature_column_map)) + " but got " +
                std::to_string(x.size(-1))};
    }
    if ((m_column_base < 0) || (m_column_qual < 0) || (m_column_strand < 0) ||
        (m_column_mapq < 0)) {
        throw std::runtime_error{
                "ModelVariantPerceiver: One or more of the fixed feature indices is not "
                "valid! "
                "Indices: base = " +
                std::to_string(m_column_base) + ", qual = " + std::to_string(m_column_qual) +
                ", strand = " + std::to_string(m_column_strand) +
                ", mapq = " + std::to_string(m_column_mapq)};
    }
    if (m_use_dwells && (m_column_dwell < 0)) {
        throw std::runtime_error{
                "ModelVariantPerceiver: The dwell column index is not valid! Got: " +
                std::to_string(m_column_dwell)};
    }
    if (m_use_haplotags && (m_column_haplotag < 0)) {
        throw std::runtime_error{
                "ModelVariantPerceiver: The haplotag column index is not valid! Got: " +
                std::to_string(m_column_haplotag)};
    }
    if (m_use_snp_qv && (m_column_snp_qv < 0)) {
        throw std::runtime_error{
                "ModelVariantPerceiver: The snp_qv column index is not valid! Got: " +
                std::to_string(m_column_snp_qv)};
    }
}

at::Tensor ModelVariantPerceiver::create_embedded_features(const at::Tensor& in_x) {
    /**
     * Example:
     *      Input:  in_x.shape = 16, 300, 20, 7
     *      Return: x.shape = 16, 300, 20, 10
     *              (m_bases_embedding_size == 6, and 4 additional tensors concatenated:
     *              torch::cat(bases + strands + haptags, scaled_q_scores, scaled_mapqc, dwells, snp_qv)
     *                                      6           +       1        +      1      +    1  +    1       = 10
     */
    utils::ScopedProfileRange spr1("ModelVariantPerceiver::create_embedded_features", 2);

    LOG_TRACE_DTYPE("[ModelVariantPerceiver::create_embedded_features] Input: in_x.dtype() = {}",
                    torch::toString(in_x.scalar_type()));

    validate_feature_tensor(in_x);

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
        at::Tensor scaled_mapqc = (in_x.select(-1, m_column_mapq) / 25).add_(-1).unsqueeze(-1);
        features.emplace_back(std::move(scaled_mapqc));
    }

    if (m_use_dwells) {
        at::Tensor dwells = in_x.select(-1, m_column_dwell).unsqueeze(-1);
        features.emplace_back(std::move(dwells));
    }

    if (m_use_snp_qv) {
        at::Tensor snp_qv = (in_x.select(-1, m_column_snp_qv) / 25).add_(-1).unsqueeze(-1);
        features.emplace_back(std::move(snp_qv));
    }

    at::Tensor x = torch::cat(features, -1);

    LOG_TRACE_DTYPE("[ModelVariantPerceiver::create_embedded_features] Output: x.dtype() = {}",
                    torch::toString(x.scalar_type()));

    return x;
}

at::Tensor ModelVariantPerceiver::forward_impl(const at::Tensor& in_x) {
    utils::ScopedProfileRange spr1("ModelVariantPerceiver::forward_impl", 1);

    LOG_TRACE("[ModelVariantPerceiver::forward_impl] in_x.shape = {}",
              utils::tensor_shape_as_string(in_x));

    LOG_TRACE_DTYPE("[ModelVariantPerceiver::forward_impl] Input: in_x.dtype() = {}",
                    torch::toString(in_x.scalar_type()));

    at::Tensor x = create_embedded_features(in_x);

    LOG_TRACE_DTYPE("[ModelVariantPerceiver::forward_impl] x.dtype() = {}",
                    torch::toString(x.scalar_type()));

    x = x.permute({0, 2, 3, 1});  // batch x reads x features x positions

    const int64_t b = x.size(0);  // Batch size
    const int64_t d = x.size(1);
    // const int64_t f = x.size(2);
    const int64_t p = x.size(3);  // Num positions

    x = x.flatten(0, 1);
    x = m_read_level_conv(x);  // b*d x cnn_size x p
    x = x.view({b, d, -1, p});

    x = x.permute({0, 3, 1, 2});

    at::Tensor reads = m_expansion_layer(x);

    at::Tensor haplotype_sequence =
            m_latent_init.unsqueeze(0)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand({b, p, -1, -1})
                    .to(reads.device());  // (batch_size, num_positions, dimension)

    for (auto& layer : *m_blocks) {
        std::tie(reads, haplotype_sequence) =
                layer->as<MessagePassingBlock>()->forward(reads, haplotype_sequence);
    }

    haplotype_sequence.squeeze_(2);

    if (m_use_decoder_lstm) {
        haplotype_sequence = std::get<0>(m_decoder_lstm(haplotype_sequence));
    } else {
        haplotype_sequence = m_decoder_identity(haplotype_sequence);
    }

    at::Tensor out = m_output(haplotype_sequence).view({b, p, m_ploidy, m_num_classes});

    LOG_TRACE_DTYPE("[ModelVariantPerceiver::forward_impl] Output: out.dtype() = {}",
                    torch::toString(out.scalar_type()));

    return out;
}

}  // namespace dorado::secondary
