#include "remora_scaler.h"

#include "remora_utils.h"
#include "utils/math_utils.h"

#include <algorithm>
#include <iterator>

RemoraScaler::RemoraScaler(const std::vector<float>& kmer_levels,
                           size_t kmer_len,
                           size_t centre_index)
        : m_kmer_levels(kmer_levels), m_kmer_len(kmer_len), m_centre_index(centre_index) {
    // ensure that the levels were the length we expected
    assert(m_kmer_levels.size() == static_cast<size_t>(1 << (2 * m_kmer_len)));
}

size_t RemoraScaler::index_from_int_kmer(const int* int_kmer_start, size_t kmer_len) {
    size_t index = 0;
    for (int kmer_pos = 0; kmer_pos < static_cast<int>(kmer_len); ++kmer_pos) {
        index += *(int_kmer_start + kmer_len - kmer_pos - 1) * (1 << (2 * kmer_pos));
    }
    return index;
}

std::vector<float> RemoraScaler::extract_levels(const std::vector<int>& int_seq) const {
    std::vector<float> levels(int_seq.size(), 0.f);
    if (int_seq.size() < m_kmer_len) {
        return levels;
    }

    auto int_kmer_start = int_seq.data();
    for (size_t pos = 0; pos < int_seq.size() - m_kmer_len; ++pos) {
        levels[pos + m_centre_index] =
                m_kmer_levels[index_from_int_kmer(int_kmer_start + pos, m_kmer_len)];
    }
    return levels;
}

std::pair<float, float> RemoraScaler::rescale(const torch::Tensor samples,
                                              const std::vector<uint64_t>& seq_to_sig_map,
                                              const std::vector<float>& levels,
                                              size_t clip_bases) const {
    if (m_kmer_levels.empty()) {
        return {0.f, 1.f};
    }

    // do calc and scaling

    std::vector<float> optim_dacs;
    std::vector<float> new_levels;
    // get the mid-point of the base
    std::transform(std::next(std::begin(seq_to_sig_map)), std::end(seq_to_sig_map),
                   std::begin(seq_to_sig_map), std::back_inserter(optim_dacs),
                   [&samples](auto first_pos, auto second_pos) {
                       auto pos = (first_pos + second_pos) / 2;
                       if (pos < samples.size(0)) {
                           return samples[pos].item().toFloat();
                       } else {
                           return 0.f;
                       }
                   });

    if (clip_bases > 0 && levels.size() > clip_bases * 2) {
        new_levels = {std::begin(levels) + clip_bases, std::end(levels) - clip_bases};
        optim_dacs = {std::begin(optim_dacs) + clip_bases, std::end(optim_dacs) - clip_bases};
    } else {
        new_levels = levels;
    }

    std::vector<float> quants(19);
    std::generate(std::begin(quants), std::end(quants), [n = 0.f]() mutable { return n += 0.05f; });
    new_levels = ::utils::quantiles(new_levels, quants);
    optim_dacs = ::utils::quantiles(optim_dacs, quants);

    auto [new_scale, new_offset, rcoeff] = ::utils::linear_regression(optim_dacs, new_levels);
    return {new_offset, new_scale};
}
