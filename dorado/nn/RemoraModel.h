#pragma once

#include <torch/torch.h>

#include <string>
#include <vector>

namespace utils {
struct BaseModInfo;
}

torch::nn::ModuleHolder<torch::nn::AnyModule> load_remora_model(const std::string& path,
                                                                torch::TensorOptions options);

/// Helper struct for storing base modification results.
struct BaseModStats {
    size_t num_states;             ///< The number of potential states per sequence postion.
    torch::Tensor base_mod_probs;  ///< The modified base likelihoods.
};

struct BaseModParams {
    std::vector<std::string> mod_long_names;  ///< The long names of the modified bases.
    std::string motif;                        ///< The motif to look for modified bases within.
    size_t base_mod_count;                    ///< The number of modifications for the base.
    size_t motif_offset;  ///< The position of the canonical base within the motif.
    size_t context_before;  ///< The number of context samples in the signal the network looks at around a candidate base.
    size_t context_after;  ///< The number of context samples in the signal the network looks at around a candidate base.
    int bases_before;  ///< The number of bases before the primary base of a kmer.
    int bases_after;   ///< The number of bases after the primary base of a kmer.
    int offset;
    std::string mod_bases;
    std::vector<float> refine_kmer_levels;  ///< Expected kmer levels for rough rescaling
    size_t refine_kmer_len;                 ///< Length of the kmers for the specified kmer_levels
    size_t refine_kmer_center_idx;  ///< The position in the kmer at which to check the levels
    bool refine_do_rough_rescale;   ///< Whether to perform rough rescaling
};

class RemoraCaller {
    constexpr static torch::ScalarType dtype = torch::kFloat32;
    torch::nn::ModuleHolder<torch::nn::AnyModule> m_module{nullptr};
    torch::TensorOptions m_options;
    torch::Tensor m_input_sigs;
    torch::Tensor m_input_seqs;

    BaseModParams m_params;
    const int m_batch_size;

    std::vector<size_t> get_motif_hits(const std::string& seq) const;

public:
    RemoraCaller(const std::string& model, const std::string& device, int batch_size = 1000);
    std::pair<torch::Tensor, std::vector<size_t>> call(torch::Tensor signal,
                                                       const std::string& seq,
                                                       const std::vector<uint8_t>& moves);
    const BaseModParams& params() const { return m_params; }
};

class RemoraRunner {
    using BaseModInfo = ::utils::BaseModInfo;

    // one caller per model
    std::vector<std::shared_ptr<RemoraCaller>> m_callers;
    std::vector<size_t> m_base_prob_offsets;
    size_t m_num_states;

    std::shared_ptr<BaseModInfo> m_base_mod_info;

public:
    RemoraRunner(const std::vector<std::string>& model_paths, const std::string& device);
    BaseModStats run(torch::Tensor signal,
                     const std::string& seq,
                     const std::vector<uint8_t>& moves);

    std::shared_ptr<const BaseModInfo> base_mod_info() const {
        return std::const_pointer_cast<const BaseModInfo>(m_base_mod_info);
    }
};

class RemoraEncoder {
private:
    std::vector<float> m_encoded_data;
    std::vector<int> m_sample_offsets;
    int m_bases_before;
    int m_kmer_len;
    int m_padding;
    int m_block_stride;
    int m_context_blocks;
    int m_seq_len;
    int m_signal_len;

    int compute_sample_pos(int base_pos) const;

public:
    /** Encoder for Remora-style modified base detection.
     *  @param block_stride The number of samples corresponding to a single entry in the movement vector.
     *  @param context_blocks The number of blocks corresponding to a slice of encoded data.
     *  @param bases_before The number of bases before the primary base of each kmer.
     *  @param bases_after The number of bases after the primary base of each kmer.
     */
    RemoraEncoder(size_t block_stride, size_t context_blocks, int bases_before, int bases_after);

    /** Encode sequence data for input to modified base detection network.
     *  @param moves The movement vector from the basecall.
     *  @param sequence The called sequence.
     *
     *  The length of the encoded data vector is equal to:
     *    (moves.size() + 2 * padding) * block_stride * kmer_len * 4,
     *  where
     *    kmer_len = bases_before + bases_after + 1
     *  and
     *    padding = total_slice_blocks / 2
     *
     *  This provides a 1-hot encoding of the kmer corresponding to each data sample. So a sample corresponding to the
     *  kmer ATC would be encoded as:
     *
     *  [ 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0 ]
     *
     *  This would be repeated for each sample corresponding to that kmer. So if the basecall stayed on that kmer through
     *  3 blocks, with a block-strike of 2, it would be repeated 6 times. The number of times the basecall stayed on the
     *  primary base of the kmer determines the number of repeats. So if, in the above example, the middle base is the
     *  primary one (bases_before = bases_after = 1), and the move vector indicates that the basecall stayed twice after
     *  emitting the T, then with a block_stride of 2 that would mean the kmer would be repeated 6 times.
     *
     *  The padding value is the number of additional blocks of data that we should pretend exist before and after the
     *  signal corresponding to the basecalled sequence. The function will behave as though each of these blocks
     *  corresponds to an N being emitted in the sequence. Kmers at the beginning and end will therefore have some N's in
     *  them, which are encoded as all zeros.
     */
    void encode_remora_data(const std::vector<uint8_t>& moves, const std::string& sequence);

    /// Get the full encoded data vector.
    const std::vector<float>& get_encoded_data() const { return m_encoded_data; }

    /// Get the sample offsets for the sequence.
    const std::vector<int>& get_sample_offsets() const { return m_sample_offsets; }

    /// Helper structure for specifying the context and returning the corresponding encoded data.
    struct Context {
        const float* data;    ///< Pointer to encoded data slice.
        size_t size;          ///< Size of encoded data slice.
        size_t first_sample;  ///< Index of first raw data sample for the slice.
        size_t num_samples;   ///< Number of samples of raw data in the slice.
        size_t lead_samples_needed;  ///< Number of samples, if any, to pad the beginning of the raw data slice with.
        size_t tail_samples_needed;  ///< Number of samples, if any, to pad the end of the raw data slice with.
    };

    /** Get the encoded data of the context centered on a specified sequence position.
     *  @param seq_pos The position of the base to center the encoded data on.
     *  @return Encoded data for the context.
     *
     *  The returned context will contain kmer_len * 4 entries for each sample position. The total number of sample
     *  positions is given by N = slice_blocks * block_stride. The context will be aligned so that sample N/2
     *  is the middle sample corresponding to the kmer in which the specified base is the primary base.
     */
    Context get_context(size_t seq_pos) const;
};

/// Calculates new scaling values for improved modified base detection
class RemoraScaler {
private:
    const std::vector<float>& m_kmer_levels;
    const size_t m_kmer_len;
    const size_t m_centre_index;

    /** Get the index in the model kmer_levels corresponding to the input kmer
     *  @param int_kmer_start Pointer to the start of the kmer in the sequence
     *  @param kmer_len The length of the kmer
     *  @return The index of the kmer
     */
    static inline size_t index_from_int_kmer(const int* int_kmer_start, size_t kmer_len);

public:
    /// Convert sequence to ints
    static std::vector<int> seq_to_ints(const std::string& sequence);

    /// Convert move table to vector of indices
    static std::vector<uint64_t> moves_to_map(const std::vector<uint8_t>& moves,
                                              size_t block_stride,
                                              size_t signal_len);

    /** Get the expected normalized daq levels for in the input basecall sequence.
     *  @param int_seq The basecall sequence, encoded as integers with A=0, C=1, G=2, T=3
     *  @return A vector of the expected normalized daq level for each base
     */
    std::vector<float> extract_levels(const std::vector<int>& int_seq) const;

    /** Calculate the new offset and scale 
     *  @param samples The normalized samples for the basecalled sequence
     *  @param seq_to_sig_map The indices of the samples corresponding to moves in the move table
     *  @param levels The expected levels for each kmer in the basecalled sequence
     *  @param clip_bases The number of bases to trim from the start and end of the sequence
     * 
     *  @return The new offset and scale values
     */
    std::pair<float, float> rescale(const torch::Tensor samples,
                                    const std::vector<uint64_t>& seq_to_sig_map,
                                    const std::vector<float>& levels,
                                    size_t clip_bases = 10) const;

    /** Scale calculator for v1 Remora-style modified base detection.
     *  @param kmer_levels A vector of expected signal levels per kmer.
     *  @param kmer_len The length of the kmers referred to in kmer_levels.
     *  @param centre_index The position in the kmer at which to set the levels.
     */
    RemoraScaler(const std::vector<float>& kmer_levels, size_t kmer_len, size_t centre_index);
};
