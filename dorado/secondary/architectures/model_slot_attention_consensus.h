#pragma once

#include "model_latent_space_lstm.h"
#include "model_torch_base.h"

#include <ATen/ATen.h>
#include <torch/nn/modules/embedding.h>
#include <torch/nn/modules/normalization.h>
#include <torch/nn/modules/rnn.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dorado::secondary {

class SlotAttentionImpl : public torch::nn::Module {
public:
    SlotAttentionImpl(int32_t num_slots,
                      int32_t dim,
                      int32_t iters,
                      float epsilon,
                      int32_t hidden_dim);

    std::pair<at::Tensor, at::Tensor> forward(at::Tensor x, at::Tensor mask, int32_t num_slots);

private:
    int32_t m_num_slots{0};
    int32_t m_dim{0};
    int32_t m_iters{3};
    float m_epsilon{1e-8f};
    int32_t m_hidden_dim{128};
    int64_t m_seed{42};
    float m_scale{0.0f};

    torch::Tensor m_slots_mu{nullptr};
    torch::Tensor m_slots_logsigma{nullptr};
    torch::nn::Linear m_to_q{nullptr};
    torch::nn::Linear m_to_k{nullptr};
    torch::nn::Linear m_to_v{nullptr};
    torch::nn::GRUCell m_gru{nullptr};
    torch::nn::Sequential m_mlp{nullptr};
    torch::nn::LayerNorm m_norm_input{nullptr};
    torch::nn::LayerNorm m_norm_slots{nullptr};
    torch::nn::LayerNorm m_norm_pre_ff{nullptr};
    torch::Tensor m_fixed_noise;
};
TORCH_MODULE(SlotAttention);

class ModelSlotAttentionConsensus : public ModelTorchBase {
public:
    ModelSlotAttentionConsensus(int32_t num_slots,
                                int32_t classes_per_slot,
                                int32_t read_embedding_size,
                                int32_t cnn_size,
                                const std::vector<int32_t>& kernel_sizes,
                                const std::string& pooler_type,
                                const std::unordered_map<std::string, std::string>& pooler_args,
                                bool use_mapqc,
                                bool use_dwells,
                                bool use_haplotags,
                                int32_t bases_alphabet_size,
                                int32_t bases_embedding_size,
                                bool add_lstm,
                                bool use_reference);

    at::Tensor forward(at::Tensor x) override;

private:
    static constexpr int32_t MAX_HAPLOTAGS{16};

    int32_t m_num_slots{2};
    int32_t m_classes_per_slot{5};
    int32_t m_read_embedding_size{128};
    int32_t m_cnn_size{128};
    std::vector<int32_t> m_kernel_sizes{1, 17};
    std::string m_pooler_type{"mean"};
    std::unordered_map<std::string, std::string> m_pooler_args{};
    bool m_use_mapqc{false};
    bool m_use_dwells{false};
    bool m_use_haplotags{false};
    int32_t m_bases_alphabet_size{6};
    int32_t m_bases_embedding_size{6};
    bool m_add_lstm{false};
    bool m_use_reference{false};
    const bool m_normalise_before_phasing{true};

    torch::nn::Embedding m_base_embedder{nullptr};
    torch::nn::Embedding m_haplotag_embedder{nullptr};
    torch::nn::Embedding m_strand_embedder{nullptr};
    ReadLevelConv m_read_level_conv{nullptr};
    torch::nn::Linear m_expansion_layer{nullptr};
    SlotAttention m_slot_attention{nullptr};
    torch::nn::Linear m_slot_classifier{nullptr};
    torch::nn::Sequential m_lstm{nullptr};

    std::pair<at::Tensor, at::Tensor> forward_impl(const at::Tensor& in_x);
    std::pair<at::Tensor, at::Tensor> quick_phase(torch::Tensor hap_probs_unphased,
                                                  at::Tensor attn,
                                                  at::Tensor features) const;

    /// @brief KMeans clustering of x with k clusters and n_iters iterations.
    ///        Init random cluster centers by selecting k points.
    ///        Do not shuffle because we want deterministic results.
    ///        Instead, deterministic sort of X then select first K.
    ///        Cluster_points = X[torch.randperm(len(X))[:k]].
    /// @param x Tensor of input vectors.
    /// @param k K clusters.
    /// @param n_iters Number of iterations.
    /// @return Tensor of cluster labels.
    at::Tensor kmeans_cluster(at::Tensor x, int32_t k, int32_t n_iters) const;
};

}  // namespace dorado::secondary