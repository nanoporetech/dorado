#pragma once

#include <ATen/core/TensorBody.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dorado::modbase {

/// Calculates new scaling values for improved modified base detection
class ModBaseScaler {
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
     *  @param max_bases is number of bases to calculate the rescaling from
     *
     *  @return The new offset and scale values
     */
    std::pair<float, float> calc_offset_scale(const at::Tensor& samples,
                                              const std::vector<uint64_t>& seq_to_sig_map,
                                              const std::vector<float>& levels,
                                              size_t clip_bases,
                                              size_t max_bases) const;

public:
    /**
     * Scale the input signal based on the expected kmer levels of the input basecalled sequence
     * @param signal The signal for the basecalled sequence
     * @param seq_ints The basecall sequence, encoded as integers with A=0, C=1, G=2, T=3
     * @param seq_to_sig_map The indices of the samples corresponding to moves in the move table
     * @return The rescaled input signal
    */
    at::Tensor scale_signal(const at::Tensor& signal,
                            const std::vector<int>& seq_ints,
                            const std::vector<uint64_t>& seq_to_sig_map) const;

    /** Scale calculator for v1 Remora-style modified base detection.
     *  @param kmer_levels A vector of expected signal levels per kmer.
     *  @param kmer_len The length of the kmers referred to in kmer_levels.
     *  @param centre_index The position in the kmer at which to set the levels.
     */
    ModBaseScaler(const std::vector<float>& kmer_levels, size_t kmer_len, size_t centre_index);
};

}  // namespace dorado::modbase
