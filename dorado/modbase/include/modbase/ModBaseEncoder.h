#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dorado::modbase {

class ModBaseEncoder {
private:
    int m_bases_before;
    int m_bases_after;
    int m_kmer_len;
    int m_block_stride;
    int m_context_samples;
    int m_context_samples_before;

    int m_seq_len;
    int m_signal_len;
    std::vector<int> m_sequence_ints;
    std::vector<uint64_t> m_sample_offsets;

    bool m_base_start_justified;

    int sample_pos(int base_pos) const;
    int compute_sample_pos(int base_pos) const;

    std::vector<int8_t> encode_kmer(const std::vector<int>& seq,
                                    const std::vector<int>& seq_mappings) const;

public:
    /** Encoder for Remora-style modified base detection.
     *  @param block_stride The number of samples corresponding to a single entry in the movement vector.
     *  @param context_samples The number of samples corresponding to a slice of encoded data.
     *  @param bases_before The number of bases before the primary base of each kmer.
     *  @param bases_after The number of bases after the primary base of each kmer.
     *  @param base_start_justified To justifiy the kmer encoding the the start of the base
     */
    ModBaseEncoder(size_t block_stride,
                   size_t context_samples,
                   int bases_before,
                   int bases_after,
                   bool base_start_justified);

    /** Initialize the sequence and movement map from which to generate encodings
     *  @param sequence_ints The basecall sequence encoded as integers (A=0, C=1, G=2, T=3)
     *  @param seq_to_sig_map An array indicating the position in the signal at which the corresponding base begins/the 
     *  previous base ends. The final value in the array should be the length of the signal. @see ::utils::moves_to_map
     */
    void init(const std::vector<int>& sequence_ints, const std::vector<uint64_t>& seq_to_sig_map);

    /// Helper structure for specifying the context and returning the corresponding encoded data.
    struct Context {
        std::vector<int8_t> data;  ///< Encoded data slice
        size_t first_sample;       ///< Index of first raw data sample for the slice.
        size_t num_existing_samples;  ///< Number of samples of raw data in the slice that already exist.
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
     *  The data is arranged in Feature-Time order i.e each column corresponds to the kmer at a given sample.
     */
    Context get_context(size_t seq_pos) const;
};

}  // namespace dorado::modbase
