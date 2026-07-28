#pragma once

#include "flash_attention_module.h"
#include "model_latent_space_lstm.h"
#include "nn/RMSNorm.h"
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
#include <utility>
#include <vector>

#define ENABLE_LOCAL_ATTENTION_MASK 0

namespace dorado::secondary {

enum class EmbeddingType {
    ROTATIONAL,
    WRAP_LEARNED,
    IDENTITY,
};

EmbeddingType parse_embedding_type(const std::string& type);

bool parse_latent_init_from_ref(const std::string& init_method);

/**
 * \brief Rotary embedding implementation.
 *
 * NOTE: There is a very similar implementation in TxModules.cpp, but that one differs in the
 *       registered buffers and the way that it handles computing the cos/sin values
 *       (it precomputes them once in the constructor for the max seq length, then reuses them).
 */
class RotaryEmbeddingImpl : public torch::nn::Module {
public:
    RotaryEmbeddingImpl(int64_t dim,
                        float theta,
                        int64_t max_seq_len,
                        const at::TensorOptions& options);

    std::pair<at::Tensor, at::Tensor> forward(at::Tensor q, at::Tensor k);

protected:
    int64_t m_dim{0};
    float m_theta{0};
    at::Tensor m_cos_freqs{nullptr};
    at::Tensor m_sin_freqs{nullptr};

private:
    void expand_freq_dims();
};
TORCH_MODULE(RotaryEmbedding);

class AbsoluteRotaryEmbeddingImpl : public RotaryEmbeddingImpl {
public:
    AbsoluteRotaryEmbeddingImpl(int64_t dim,
                                float theta,
                                int64_t max_read_depth,
                                const at::TensorOptions& options);

    at::Tensor forward(at::Tensor x, const at::Tensor& depths);

private:
    int64_t m_max_depth{0};
    void expand_freq_dims();
};
TORCH_MODULE(AbsoluteRotaryEmbedding);

class EmbeddingWrapperImpl : public torch::nn::Module {
public:
    EmbeddingWrapperImpl(int64_t max_depth, int64_t dimension);

    at::Tensor forward(at::Tensor x, const at::Tensor& depths);

private:
    int64_t m_max_depth{0};
    torch::nn::Embedding m_embedding{nullptr};
};
TORCH_MODULE(EmbeddingWrapper);

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

class MultiHeadCrossAttentionImpl : public torch::nn::Module {
public:
    MultiHeadCrossAttentionImpl(int64_t d_model,
                                int64_t q_max_depth,
                                int64_t kv_max_depth,
                                int64_t nhead,
                                bool embed_features,
                                EmbeddingType embedding_type,
                                bool qkv_bias,
                                bool out_bias,
                                bool use_varlen,
                                const std::optional<int64_t>& rotary_dim,
                                const std::optional<int64_t>& attn_window);

    /**
     * \brief Update the `update_seq` tensor by attending to the `cross_attn_seqs` tensor.
     * \param x Tensor of shape (batch_size, num_positions, num_sequences_q, input_dim).
     * \param y Tensor of shape (batch_size, num_positions, num_sequences_kv, input_dim).
     * \param pos_mask Tensor of shape (batch_size, num_heads, num_positions * num_sequences_q, num_positions * num_sequences_kv).
     * \returns out Tensor of shape (batch_size, num_positions, num_sequences, output_dim).
     */
    at::Tensor forward(const at::Tensor& x,
                       const at::Tensor& y,
                       const at::Tensor& x_depths,
                       const at::Tensor& y_depths,
                       const std::optional<at::Tensor>& x_pos_mask,
                       const std::optional<at::Tensor>& y_pos_mask);

private:
    int64_t m_nhead{0};
    int64_t m_head_dim{0};
    EmbeddingType m_q_embedding_type{EmbeddingType::IDENTITY};
    EmbeddingType m_k_embedding_type{EmbeddingType::IDENTITY};
    std::optional<int64_t> m_rotary_dim{std::nullopt};
    std::optional<int64_t> m_attn_window{std::nullopt};

#if ENABLE_LOCAL_ATTENTION_MASK
    mutable at::Tensor m_cached_local_mask{nullptr};
    mutable c10::Device m_cached_local_mask_device{c10::kCPU};
    mutable int64_t m_cached_local_mask_t{-1};
    mutable int64_t m_cached_local_mask_num_q{-1};
    mutable int64_t m_cached_local_mask_num_kv{-1};
    mutable int64_t m_cached_local_mask_window{-1};
#endif

    const bool m_use_varlen{true};

    torch::nn::Linear m_kv_proj{nullptr};
    torch::nn::Linear m_q_proj{nullptr};
    torch::nn::Linear m_out_proj{nullptr};
    FlashAttentionModule m_flash_attention{nullptr};
    RotaryEmbedding m_positional_embeddings{nullptr};
    torch::nn::Identity m_q_embedding_ident{nullptr};
    torch::nn::Identity m_k_embedding_ident{nullptr};
    AbsoluteRotaryEmbedding m_q_embedding_rot{nullptr};
    AbsoluteRotaryEmbedding m_k_embedding_rot{nullptr};
    EmbeddingWrapper m_q_embedding_wrap{nullptr};
    EmbeddingWrapper m_k_embedding_wrap{nullptr};

#if ENABLE_LOCAL_ATTENTION_MASK
    /**
     * \brief Implements the following masking logic:
     *          abs((query_pos % T) - (key_pos % T)) <= self.attn_window
     */
    at::Tensor local_attention_mask(int64_t T,
                                    int64_t num_q_seqs,
                                    int64_t num_kv_seqs,
                                    int64_t attn_window) const;
#endif

    at::Tensor attn_fn(const at::Tensor& q,
                       const at::Tensor& k,
                       const at::Tensor& v,
                       const at::Tensor& q_depths,
                       const at::Tensor& kv_depths,
                       const std::optional<at::Tensor>& q_pos_mask,
                       const std::optional<at::Tensor>& kv_pos_mask) const;
};
TORCH_MODULE(MultiHeadCrossAttention);

class MultiSequenceCrossAttentionBlockImpl : public torch::nn::Module {
public:
    MultiSequenceCrossAttentionBlockImpl(int64_t d_model,
                                         int64_t q_max_depth,
                                         int64_t kv_max_depth,
                                         int64_t nhead,
                                         bool embed_features,
                                         EmbeddingType embedding_type,
                                         bool qkv_bias,
                                         bool out_bias,
                                         bool use_varlen,
                                         const std::optional<int64_t>& rotary_dim,
                                         const std::optional<int64_t>& attn_window,
                                         int64_t dim_feedforward,
                                         float deepnorm_alpha);

    at::Tensor forward(at::Tensor x,
                       const at::Tensor& y,
                       const at::Tensor& x_depths,
                       const at::Tensor& y_depths,
                       const std::optional<at::Tensor>& x_pos_mask,
                       const std::optional<at::Tensor>& y_pos_mask);

private:
    at::Tensor m_deepnorm_alpha{torch::empty({})};

    MultiHeadCrossAttention m_attention{nullptr};
    SwiGLU m_ff{nullptr};
    nn::RMSNorm m_norm1{nullptr};
    nn::RMSNorm m_norm2{nullptr};
};
TORCH_MODULE(MultiSequenceCrossAttentionBlock);

class SelfAttentionBlockImpl : public MultiSequenceCrossAttentionBlockImpl {
public:
    SelfAttentionBlockImpl(int64_t d_model,
                           int64_t max_depth,
                           int64_t nhead,
                           bool embed_features,
                           EmbeddingType embedding_type,
                           bool qkv_bias,
                           bool out_bias,
                           const std::optional<int64_t>& rotary_dim,
                           const std::optional<int64_t>& attn_window,
                           int64_t dim_feedforward,
                           float deepnorm_alpha);

    at::Tensor forward(const at::Tensor& x, const at::Tensor& depth);
};
TORCH_MODULE(SelfAttentionBlock);

class MessagePassingBlockImpl : public torch::nn::Module {
public:
    MessagePassingBlockImpl(int64_t dim,
                            int64_t read_max_dim,
                            int64_t num_heads,
                            int64_t self_attn_layers_per_block,
                            bool embed_features,
                            EmbeddingType embedding_type,
                            bool update_read_embeddings,
                            bool cross_attend_read_embeddings,
                            const std::optional<int64_t>& attn_window);

    /**
     * \brief Forward function of the MessagePassingBlock module.
     * \param read_seqs Tensor of shape (total_depth, num_positions, dim).
     * \param hap_seqs Tensor of shape (batch_size, num_positions, dim).
     * \param read_depths Tensor containing number of read rows for each sample in the batch, shape (batch_size).
     * \param hap_depths Tensor containing number of haplotype embedding rows for each sample in the batch, shape (batch_size).
     * \param pos_mask Bool tensor with false at positions in read_seqs that should be masked, shape (total_depth, num_positions).
     * \return Updated read_seqs, shape (total_depth, num_positions, dim)
     *         Updated hap_seqs, shape (batch_size, num_positions, dim).
     */
    std::pair<at::Tensor, at::Tensor> forward(at::Tensor read_seqs,
                                              at::Tensor hap_seqs,
                                              const at::Tensor& read_depths,
                                              const at::Tensor& hap_depths,
                                              const at::Tensor& pos_mask);

private:
    bool m_update_read_embeddings{false};
    bool m_cross_attend_read_embeddings{false};
    MultiSequenceCrossAttentionBlock m_reads_to_haplotypes{nullptr};
    torch::nn::ModuleList m_haplotype_self_attention{nullptr};
    MultiSequenceCrossAttentionBlock m_haplotypes_to_reads{nullptr};
};
TORCH_MODULE(MessagePassingBlock);

class ModelVariantPerceiver : public ModelTorchBase {
public:
    ModelVariantPerceiver(const MustConstructWithFactory& ctor_tag,
                          int32_t read_max_depth,
                          int32_t ploidy,
                          int32_t num_classes,
                          int32_t cnn_size,
                          const std::vector<int32_t>& kernel_sizes,
                          int32_t dimension,
                          int32_t num_blocks,
                          int32_t num_heads,
                          int32_t self_attn_layers_per_block,
                          bool use_mapqc,
                          bool use_dwells,
                          bool use_haplotags,
                          bool use_snp_qv,
                          int32_t bases_alphabet_size,
                          int32_t bases_embedding_size,
                          // bool time_steps,
                          bool use_decoder_lstm,
                          bool use_per_read_embedding,
                          EmbeddingType embedding_type,
                          bool update_read_embeddings,
                          // std::optional<int32_t> attn_window,
                          bool latent_ref_init,
                          const FeatureColumnMap& feature_column_map);

    /**
     * \brief Forward pass.
     * \param x Read level feature matrix, shape
     *          (num_batch, num_positions, num_reads (padded), num_features).
     * \param ref_seq The integer encoded haploid reference.
     *                Can be None if the model doesn't require it, else has shape
     *                (num_batch, num_positions).
     * \return Logits for positionwise predictions (num_positions, num_slots, num_classes).
     */
    at::Tensor forward(at::Tensor x) override;
    at::Tensor forward(const at::Tensor& x, const std::optional<at::Tensor>& ref_seqs);
    /**
     * \brief Predict on a batch with device and precision handling.
     */
    at::Tensor predict_on_device_batch(const BatchedData& batched_data) override;

    double estimate_batch_memory(const std::vector<int64_t>& batch_tensor_shape) const override;

    bool requires_ref(void) const override;

private:
    static constexpr int32_t MAX_HAPLOTAGS{16};

    int32_t m_ploidy{2};
    int32_t m_num_classes{5};
    int32_t m_cnn_size{128};
    std::vector<int32_t> m_kernel_sizes{1, 17};
    int32_t m_dimension{256};
    int32_t m_num_blocks{4};
    bool m_use_mapqc{false};
    bool m_use_dwells{false};
    bool m_use_haplotags{false};
    bool m_use_snp_qv{false};
    bool m_latent_ref_init{false};
    int32_t m_bases_alphabet_size{6};
    int32_t m_bases_embedding_size{6};
    bool m_use_decoder_lstm{false};
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
    torch::nn::Linear m_latent_ref_project{nullptr};

    int32_t m_column_base{-1};
    int32_t m_column_qual{-1};
    int32_t m_column_strand{-1};
    int32_t m_column_mapq{-1};
    int32_t m_column_dwell{-1};
    int32_t m_column_haplotag{-1};
    int32_t m_column_snp_qv{-1};

    void validate_feature_tensor(const at::Tensor& x) const;

    /**
     * \brief Preprocessing of input tensor.
     * \param in_x Tensor of shape (batch_size, num_positions, num_sequences, num_features).
     * \return embedding tensor of shape (total_depth, num_positions, dim),
               mask tensor of shape (total_depth, num_positions).
               batch depths tensor of shape (batch_size).
     */
    std::tuple<at::Tensor, at::Tensor, at::Tensor> create_embedded_features(const at::Tensor& in_x);

    at::Tensor create_latent_embedding(const std::optional<at::Tensor>& ref_seq,
                                       int64_t N,
                                       int64_t T);

    at::Tensor forward_impl(const at::Tensor& x, const std::optional<at::Tensor>& ref_seqs);
};

}  // namespace dorado::secondary
