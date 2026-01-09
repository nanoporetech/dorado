#pragma once

#include "model_latent_space_lstm.h"
#include "nn/TxModules.h"
#include "secondary/architectures/model_torch_base.h"
#include "secondary/features/encoder_base.h"

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <torch/nn/modules/embedding.h>
#include <torch/nn/modules/normalization.h>
#include <torch/nn/modules/rnn.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dorado::secondary {

/**
 * \brief Rotary embedding implementation.
 *
 * NOTE: There is a very similar implementation in TxModules.cpp, but that one differs in the
 *       registered buffers and the way that it handles computing the cos/sin values
 *       (it precomputes them once in the constructor for the max seq length, then reuses them).
 */
class RotaryEmbeddingImpl : public torch::nn::Module {
public:
    RotaryEmbeddingImpl(int64_t dim, float theta, const at::TensorOptions& options);

    std::pair<at::Tensor, at::Tensor> forward(at::Tensor q, at::Tensor k);

private:
    int64_t m_dim{0};
    float m_theta{0};
    at::Tensor m_inv_freq{nullptr};

    at::Tensor rotate_half(const at::Tensor& x) const;
};
TORCH_MODULE(RotaryEmbedding);

/**
 * \brief SwiGLU implementation.
 *          There is an existing, almost compatible SwiGLU implementation from the basecaller (dorado::nn::GatedMLP),
 *          but it generates tensors of incompatible shape for our use case (higher dimensionality). TODO.
 */
class SwiGLUImpl : public torch::nn::Module {
public:
    SwiGLUImpl(int32_t in_features, int32_t hidden_features, bool bias);

    at::Tensor forward(const at::Tensor& x);

private:
    torch::nn::Linear m_fc1{nullptr};
    torch::nn::Linear m_fc2{nullptr};
};
TORCH_MODULE(SwiGLU);

class MultiSequenceCrossAttentionBlockImpl : public torch::nn::Module {
public:
    MultiSequenceCrossAttentionBlockImpl(int64_t dim,
                                         int64_t ploidy,
                                         int64_t n_pos,
                                         int64_t num_heads,
                                         int64_t max_depth,
                                         float dropout,
                                         bool qkv_bias,
                                         const std::optional<int64_t>& attn_window);

    /**
     * \brief Update the `update_seq` tensor by attending to the `cross_attn_seqs` tensor.
     * \param x Tensor of shape (batch_size, num_positions, num_sequences, input_dim).
     * \param cross_attn_seq Tensor of shape (batch_size, num_positions, num_sequences, input_dim).
     * \returns out Tensor of shape (batch_size, num_positions, num_sequences, output_dim).
     */
    at::Tensor forward(at::Tensor x, const at::Tensor& cross_attn_seqs);

private:
    int64_t m_num_heads{0};
    int64_t m_head_dim{0};
    std::optional<int64_t> m_attn_window{std::nullopt};

    torch::nn::Linear m_kv_proj{nullptr};
    torch::nn::Linear m_q_proj{nullptr};
    torch::nn::Embedding m_read_embeddings{nullptr};
    RotaryEmbedding m_positional_embeddings{nullptr};
    SwiGLU m_out_proj{nullptr};
    nn::RMSNorm m_norm1{nullptr};
    nn::RMSNorm m_norm2{nullptr};
    // torch::nn::Dropout m_attn_dropout{nullptr};

    /**
     * \brief Implements the following masking logic:
     *          abs((query_pos % T) - (key_pos % T)) <= self.attn_window
     *
     * TODO: Cache results of the mask to avoid recomputation.
     */
    at::Tensor local_attention_mask(const int64_t T,
                                    const int64_t num_q_seqs,
                                    const int64_t num_kv_seqs,
                                    const int64_t attn_window) const;

    at::Tensor attn_fn(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) const;
};
TORCH_MODULE(MultiSequenceCrossAttentionBlock);

class SelfAttentionBlockImpl : public torch::nn::Module {
public:
    SelfAttentionBlockImpl(int64_t dim,
                           int64_t num_heads,
                           float dropout,
                           const std::optional<int64_t>& attn_window);

    at::Tensor forward(const at::Tensor& x);

private:
    MultiSequenceCrossAttentionBlock m_self_attention{nullptr};
    nn::RMSNorm m_norm{nullptr};
};
TORCH_MODULE(SelfAttentionBlock);

class MessagePassingBlockImpl : public torch::nn::Module {
public:
    MessagePassingBlockImpl(int64_t dim,
                            int64_t num_heads,
                            float dropout,
                            bool update_read_embeddings,
                            bool cross_attend_read_embeddings,
                            const std::optional<int64_t>& attn_window);

    /**
     * \brief Forward function of the MessagePassingBlock module.
     * \param read_seqs Tensor of shape (batch_size, num_positions, num_sequences, dim).
     * \param hap_seqs Tensor of shape (batch_size, num_positions, num_sequences, dim).
     * \return out Tensor of shape (batch_size, num_positions, num_sequences, dim).
     */
    std::pair<at::Tensor, at::Tensor> forward(at::Tensor read_seqs, at::Tensor hap_seqs);

private:
    bool m_update_read_embeddings{false};
    bool m_cross_attend_read_embeddings{false};
    MultiSequenceCrossAttentionBlock m_reads_to_haplotypes{nullptr};
    SelfAttentionBlock m_haplotype_self_attention{nullptr};
    MultiSequenceCrossAttentionBlock m_haplotypes_to_reads{nullptr};

    nn::RMSNorm m_norm_1{nullptr};
    nn::RMSNorm m_norm_2{nullptr};
};
TORCH_MODULE(MessagePassingBlock);

class ModelVariantPerceiver : public ModelTorchBase {
public:
    ModelVariantPerceiver(const MustConstructWithFactory& ctor_tag,
                          int32_t ploidy,
                          int32_t num_classes,
                          int32_t read_embedding_size,
                          int32_t cnn_size,
                          const std::vector<int32_t>& kernel_sizes,
                          int32_t dimension,
                          int32_t num_blocks,
                          int32_t num_heads,
                          bool use_mapqc,
                          bool use_dwells,
                          bool use_haplotags,
                          bool use_snp_qv,
                          int32_t bases_alphabet_size,
                          int32_t bases_embedding_size,
                          // bool time_steps,
                          bool use_decoder_lstm,
                          bool update_read_embeddings,
                          const FeatureColumnMap& feature_column_map);

    /**
     * \brief Forward pass.
     * \param x Read level feature matrix, shape
     *          (num_batch, num_positions, num_reads (padded), num_features).
     * \param ref_seq The integer encoded haploid reference.
     *                  Can be None if the model doesn't require it, else has shape
     *                  (num_batch, num_positions).
     * \return Logits for positionwise predictions (num_positions, num_slots, num_classes).
     */
    at::Tensor forward(at::Tensor x) override;

    double estimate_batch_memory(const std::vector<int64_t>& batch_tensor_shape) const override;

private:
    static constexpr int32_t MAX_HAPLOTAGS{16};

    int32_t m_ploidy{2};
    int32_t m_num_classes{5};
    int32_t m_read_embedding_size{128};
    int32_t m_cnn_size{128};
    std::vector<int32_t> m_kernel_sizes{1, 17};
    int32_t m_dimension{256};
    int32_t m_num_blocks{4};
    int32_t m_num_heads{8};
    bool m_use_mapqc{false};
    bool m_use_dwells{false};
    bool m_use_haplotags{false};
    bool m_use_snp_qv{false};
    int32_t m_bases_alphabet_size{6};
    int32_t m_bases_embedding_size{6};
    bool m_use_decoder_lstm{false};
    bool m_update_read_embeddings{false};
    FeatureColumnMap m_feature_column_map{};

    torch::nn::Embedding m_base_embedder{nullptr};
    torch::nn::Embedding m_haplotag_embedder{nullptr};
    torch::nn::Embedding m_strand_embedder{nullptr};
    ReadLevelConv m_read_level_conv{nullptr};
    torch::nn::Linear m_expansion_layer{nullptr};
    at::Tensor m_latent_init{nullptr};
    torch::nn::ModuleList m_blocks{nullptr};
    torch::nn::LSTM m_decoder_lstm{nullptr};
    torch::nn::Identity m_decoder_identity{nullptr};
    torch::nn::Linear m_output{nullptr};

    int32_t m_column_base{-1};
    int32_t m_column_qual{-1};
    int32_t m_column_strand{-1};
    int32_t m_column_mapq{-1};
    int32_t m_column_dwell{-1};
    int32_t m_column_haplotag{-1};
    int32_t m_column_snp_qv{-1};

    void validate_feature_tensor(const at::Tensor& x) const;

    at::Tensor create_embedded_features(const at::Tensor& in_x);

    at::Tensor forward_impl(const at::Tensor& x);
};

}  // namespace dorado::secondary
