#pragma once

#include <torch/torch.h>

#include <string>
#include <vector>

class RemoraEncoder {
private:
    std::vector<int> m_sample_offsets;
    int m_bases_before;
    int m_bases_after;
    int m_kmer_len;
    int m_block_stride;
    int m_context_samples;
    int m_seq_len;
    int m_signal_len;
    std::vector<int> m_sequence_ints;

    std::vector<float> m_buffer;

    int compute_sample_pos(int base_pos) const;

public:
    /** Encoder for Remora-style modified base detection.
     *  @param block_stride The number of samples corresponding to a single entry in the movement vector.
     *  @param context_samples The number of samples corresponding to a slice of encoded data.
     *  @param bases_before The number of bases before the primary base of each kmer.
     *  @param bases_after The number of bases after the primary base of each kmer.
     */
    RemoraEncoder(size_t block_stride, size_t context_samples, int bases_before, int bases_after);

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
     *  3 blocks, with a block-stride of 2, it would be repeated 6 times. The number of times the basecall stayed on the
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

    /// Get the sample offsets for the sequence.
    const std::vector<int>& get_sample_offsets() const { return m_sample_offsets; }

    /// Helper structure for specifying the context and returning the corresponding encoded data.
    struct Context {
        std::vector<float> data;  ///< Encoded data slice
        size_t size;              ///< Size of encoded data slice.
        size_t first_sample;      ///< Index of first raw data sample for the slice.
        size_t num_samples;       ///< Number of samples of raw data in the slice.
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
