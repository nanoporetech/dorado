#include "ModbaseScaler.h"

#include "utils/math_utils.h"

#include <ATen/Functions.h>
#include <ATen/TensorOperators.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <iterator>
#include <stdexcept>

namespace dorado::modbase {

ModBaseScaler::ModBaseScaler(const std::vector<float>& kmer_levels,
                             size_t kmer_len,
                             size_t centre_index)
        : m_kmer_levels(kmer_levels), m_kmer_len(kmer_len), m_centre_index(centre_index) {
    // ensure that the levels were the length we expected
    if (m_kmer_levels.size() != static_cast<size_t>(1ull << (2 * m_kmer_len))) {
        throw std::runtime_error(
                "Modbase refinement levels have invalid size for given kmer length.");
    }
}

size_t ModBaseScaler::index_from_int_kmer(const int* int_kmer_start, size_t kmer_len) {
    size_t index = 0;
    for (int kmer_pos = 0; kmer_pos < static_cast<int>(kmer_len); ++kmer_pos) {
        index += *(int_kmer_start + kmer_len - kmer_pos - 1) * (1 << (2 * kmer_pos));
    }
    return index;
}

at::Tensor ModBaseScaler::scale_signal(const at::Tensor& signal,
                                       const std::vector<int>& seq_ints,
                                       const std::vector<uint64_t>& seq_to_sig_map) const {
    NVTX3_FUNC_RANGE();
    auto levels = extract_levels(seq_ints);

    // generate the signal values at the centre of each base, create the nx5% quantiles (sorted)
    // and perform a linear regression against the expected kmer levels to generate a new shift and scale
    auto [offset, scale] = calc_offset_scale(signal, seq_to_sig_map, levels, 10, 1000);
    auto scaled_signal = signal * scale + offset;
    return scaled_signal;
}

std::vector<float> ModBaseScaler::extract_levels(const std::vector<int>& int_seq) const {
    std::vector<float> levels(int_seq.size(), 0.f);
    if (int_seq.size() < m_kmer_len) {
        return levels;
    }

    auto int_kmer_start_ptr = int_seq.data();
    auto levels_ptr = levels.data() + m_centre_index;
    for (size_t pos = 0; pos < int_seq.size() - m_kmer_len;
         ++pos, ++int_kmer_start_ptr, ++levels_ptr) {
        *(levels_ptr) = m_kmer_levels[index_from_int_kmer(int_kmer_start_ptr, m_kmer_len)];
    }
    return levels;
}

std::pair<float, float> ModBaseScaler::calc_offset_scale(
        const at::Tensor& samples,
        const std::vector<uint64_t>& seq_to_sig_map,
        const std::vector<float>& levels,
        size_t clip_bases,
        size_t max_bases) const {
    NVTX3_FUNC_RANGE();
    if (m_kmer_levels.empty()) {
        return {0.f, 1.f};
    }

    auto n = std::min({seq_to_sig_map.size() - 1, max_bases});

    std::vector<float> optim_dacs(n, 0.f);
    std::vector<float> new_levels(n, 0.f);

    {
        nvtx3::scoped_range loop{"initialize_vectors"};
        assert(samples.is_contiguous());
        assert(samples.dtype() == at::kHalf);
        using SignalType = c10::Half;
        SignalType* samples_ptr = samples.data_ptr<SignalType>();
        // get the mid-point of the base
        for (size_t i = 0; i < n; i++) {
            int pos = int((seq_to_sig_map[i] + seq_to_sig_map[i + 1]) / 2);
            optim_dacs[i] = static_cast<float>(samples_ptr[pos]);
            new_levels[i] = levels[i];
        }
    }
    if (clip_bases > 0 && levels.size() > clip_bases * 2) {
        new_levels = {std::begin(new_levels) + clip_bases, std::end(new_levels) - clip_bases};
        optim_dacs = {std::begin(optim_dacs) + clip_bases, std::end(optim_dacs) - clip_bases};
    }

    std::vector<float> quants(19);
    std::generate(std::begin(quants), std::end(quants), [i = 0.f]() mutable { return i += 0.05f; });

    new_levels = utils::quantiles(new_levels, quants);
    optim_dacs = utils::quantiles(optim_dacs, quants);

    auto [new_scale, new_offset, rcoeff] = utils::linear_regression(optim_dacs, new_levels);
    return {new_offset, new_scale};
}

}  // namespace dorado::modbase
