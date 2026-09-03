#include "splitter_utils.h"

#include "read_pipeline/base/messages/SimplexRead.h"
#include "read_pipeline/base/read_utils.h"
#include "utils/time_utils.h"
#include "utils/types.h"  // required for ModBaseInfo

#include <ATen/TensorIndexing.h>

#include <bit>
#include <type_traits>

// MSVC doesn't define __SSE2__ so we have to do it ourselves.
#if defined(_M_AMD64)
#define __SSE2__
#define __m128i_u __m128i  // MSVC doesn't provide an unaligned typedef
#endif

#if defined(__SSE2__)
#include <immintrin.h>
#elif defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace dorado::splitter {

SimplexReadPtr subread(const SimplexRead& read,
                       std::optional<PosRange> seq_range,
                       std::pair<uint64_t, uint64_t> signal_range) {
    auto subread = utils::shallow_copy_read(read);

    // Unable to determine how the events were distributed, so zero out
    subread->read_common.num_minknow_events = 0;

    subread->read_common.raw_data = subread->read_common.raw_data.index(
            {at::indexing::Slice(signal_range.first, signal_range.second)});
    subread->read_common.attributes.read_number = -1;

    //we adjust for it in new start time
    subread->read_common.split_point =
            uint32_t(read.read_common.split_point + read.read_common.num_trimmed_samples +
                     signal_range.first);
    subread->read_common.attributes.num_samples = signal_range.second - signal_range.first;
    subread->read_common.num_trimmed_samples = 0;
    subread->start_sample =
            read.start_sample + read.read_common.num_trimmed_samples + signal_range.first;
    subread->end_sample = subread->start_sample + subread->read_common.attributes.num_samples;

    auto start_time_ms = static_cast<uint64_t>(std::round(
            subread->start_sample * 1000. / subread->read_common.attributes.sample_rate));
    subread->read_common.attributes.start_time = utils::get_string_timestamp_from_unix_time_ms(
            read.run_acquisition_start_time_ms + start_time_ms);
    subread->read_common.start_time_ms = start_time_ms;
    subread->read_common.attributes.end_reason =
            "unknown";  // TODO: do we know the reason for split reads? signal_positive?
    subread->read_common.attributes.is_end_reason_mux_change = false;

    if (seq_range) {
        const int stride = read.read_common.attributes.model_stride;
        assert(signal_range.first <= signal_range.second);
        assert(signal_range.first / stride <= read.read_common.moves.size());
        assert(signal_range.second / stride <= read.read_common.moves.size());
        assert(signal_range.first % stride == 0);
        assert(signal_range.second % stride == 0 ||
               (signal_range.second == read.read_common.get_raw_data_samples() &&
                seq_range->second == read.read_common.seq.size()));

        if (seq_range->first != 0 || seq_range->second != read.read_common.seq.size()) {
            if (!read.read_common.base_mod_probs.empty()) {
                const auto num_channels = read.read_common.mod_base_info->alphabet.size();
                const auto first = read.read_common.base_mod_probs.cbegin();
                subread->read_common.base_mod_probs =
                        std::vector<uint8_t>(first + seq_range->first * num_channels,
                                             first + seq_range->second * num_channels);
            }
            if (!read.read_common.base_mod_simplex_motif_hits.empty()) {
                const auto first = read.read_common.base_mod_simplex_motif_hits.cbegin();
                subread->read_common.base_mod_simplex_motif_hits =
                        std::vector<bool>(first + seq_range->first, first + seq_range->second);
            }
        }

        subread->read_common.seq = subread->read_common.seq.substr(
                seq_range->first, seq_range->second - seq_range->first);
        subread->read_common.qstring = subread->read_common.qstring.substr(
                seq_range->first, seq_range->second - seq_range->first);
        subread->read_common.pre_trim_seq_length = subread->read_common.seq.length();
        subread->read_common.moves = std::vector<uint8_t>(
                subread->read_common.moves.begin() + signal_range.first / stride,
                subread->read_common.moves.begin() + signal_range.second / stride);
        assert(signal_range.second == read.read_common.get_raw_data_samples() ||
               subread->read_common.moves.size() * stride ==
                       subread->read_common.get_raw_data_samples());
    }

    // Initialize the subreads previous and next reads with the parent's ids.
    // These are updated at the end when all subreads are available.
    subread->prev_read = read.prev_read;
    subread->next_read = read.next_read;

    if (!read.read_common.parent_read_id.empty()) {
        subread->read_common.parent_read_id = read.read_common.parent_read_id;
    } else {
        subread->read_common.parent_read_id = read.read_common.read_id;
    }
    return subread;
}

PosRanges merge_ranges(const PosRanges& ranges, uint64_t merge_dist) {
    PosRanges merged;
    for (auto& r : ranges) {
        assert(merged.empty() || r.first >= merged.back().first);
        if (merged.empty() || r.first > merged.back().second + merge_dist) {
            merged.push_back(r);
        } else {
            merged.back().second = std::max(r.second, merged.back().second);
        }
    }
    return merged;
}

template <typename T>
SampleRanges<T> detect_pore_signal(const at::Tensor& signal,
                                   T threshold,
                                   uint64_t cluster_dist,
                                   uint64_t ignore_prefix,
                                   uint64_t ignore_spikes_threshold) {
    SampleRanges<T> clusters;

#if C10_ASAN_ENABLED
    // For some reason mold has issues finding the const version of this symbol when sanitizers are enabled.
    const auto pore_a = signal.accessor<T, 1>();
#else
    const auto pore_a = signal.accessor<const T, 1>();
#endif
    const auto pore_a_data = pore_a.data();
    const int64_t pore_a_size = pore_a.size(0);
    TORCH_CHECK(signal.stride(0) == 1, "signal should be contiguous");

    const auto over_threshold = [threshold](const T& val) {
    // ARM64 has native _Half support but x64 doesn't and has to convert a c10::Half
    // to a single precision float to compare them, so use a fast comparison since we
    // likely won't need the single precision float afterwards.
#if !defined(__aarch64__)
        if constexpr (std::is_same_v<T, c10::Half>) {
            return detail::fast_half_comparable(val) > detail::fast_half_comparable(threshold);
        }
#endif
        return val > threshold;
    };

    int64_t idx = ignore_prefix;
    while (idx < pore_a_size) {
        int64_t cl_start = -1;

        // Most of the time is spent looking for the start of a peak, so make that hot loop tight.
#if defined(__SSE2__) || defined(__ARM_NEON__)
        if constexpr (std::is_same_v<T, c10::Half>) {
#if defined(__SSE2__)
            using Register = __m128i;
            static constexpr int64_t kHalfsPerRegister = 8;
            static const Register kSignBit = _mm_set1_epi16(static_cast<short>(0x8000));
            static const Register kExpManMask = _mm_set1_epi16(0x7FFF);
            static const Register kOne = _mm_set1_epi16(1);

            const Register fast_threshold = _mm_set1_epi16(detail::fast_half_comparable(threshold));
#else
            using Register = float16x8_t;
            static constexpr int64_t kHalfsPerRegister = 8;

            const Register threshold_splat = vdupq_n_f16(threshold);
#endif

            // |idx| isn't necessarily a multiple of |kHalfsPerRegister| since we don't know where the last open pore
            // ended, so we always have to check for enough space (no (x/8)*8 trickery).
            for (; idx < pore_a_size - kHalfsPerRegister; idx += kHalfsPerRegister) {
                // Load 8 halfs into a register.
#if defined(__SSE2__)
                Register halfs =
                        _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(pore_a_data + idx));
#else
                Register halfs = vld1q_f16(reinterpret_cast<float16_t const*>(pore_a_data + idx));
#endif

#if defined(__SSE2__)
                // SIMD version of fast_half_comparable().
                Register sign = _mm_and_si128(halfs, kSignBit);
                Register em = _mm_and_si128(halfs, kExpManMask);
                // We emulate |vals = sign ? -em : em| by doing |vals = sign * em| instead.
                // TODO: if we had SSSE3 support we could use _mm_sign_epi16() instead.
                // |sign| will either be 0x8000 or 0 depending on if the sign bit is set.
                // We need to map these to -1 or 1 respectively, which we can do with a shift and a sub.
                // 1 - (0x8000 >> 14) = -1
                // 1 - (0x0000 >> 14) = 1
                Register mul = _mm_sub_epi16(kOne, _mm_srli_epi16(sign, 14));
                Register vals = _mm_mullo_epi16(mul, em);

                // over_threshold() comparison.
                const uint16_t cmp = _mm_movemask_epi8(_mm_cmpgt_epi16(vals, fast_threshold));
#else
                // Taken from here: https://developer.arm.com/community/arm-community-blogs/b/servers-and-cloud-computing-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon
                const uint16x8_t mask = vcgtq_f16(halfs, threshold_splat);
                const uint8x8_t res = vshrn_n_u16(mask, 4);
                const uint64_t cmp = vget_lane_u64(vreinterpret_u64_u8(res), 0);
#endif

                // If any of them are set then we've found a match.
                if (cmp != 0) [[unlikely]] {
#if defined(__SSE2__)
                    // Each bit in |cmp| corresponds to a byte in the register,
                    // so an int16 comparison takes 2 bits per element.
                    constexpr std::size_t shift = 1;
#else
                    // On the ARM path you get 4 bits per byte, which is 8 bits
                    // per element.
                    constexpr std::size_t shift = 3;
#endif
                    const int first_set = std::countr_zero(cmp) >> shift;
                    idx += first_set;
                    cl_start = idx;
                    break;
                }
            }
        }

        // Search the rest linearly if we haven't found one.
        if (cl_start == -1)
#endif
        {
            for (; idx < pore_a_size; idx++) {
                const T sample = pore_a_data[idx];
                if (over_threshold(sample)) {
                    cl_start = idx;
                    break;
                }
            }
        }
        if (cl_start == -1) {
            // No peak found.
            break;
        }

        // Grab the start position.
        int64_t cl_argmax = idx++;
        T cl_max = pore_a_data[cl_argmax];

        // Read until end of the peak.
        for (; idx < pore_a_size; idx++) {
            const T sample = pore_a_data[idx];
            if (!over_threshold(sample)) {
                break;
            }
            if (sample >= cl_max) {
                cl_max = sample;
                cl_argmax = idx;
            }
        }

        // Grab the end position.
        // Note that it's safe to increment |idx| here since |cl_end| is the first index
        // after the peak, and hence we don't need to test |idx| again to see if it's above
        // the threshold.
        const int64_t cl_end = idx++;

        // report cluster
        assert(cl_start < pore_a_size && cl_end <= pore_a_size);
        clusters.push_back(SampleRange(cl_start, cl_end, cl_argmax, cl_max));
    }

    // merge clusters
    SampleRanges<T> merged_clusters;
    for (auto&& cluster : clusters) {
        if (cluster.end_sample - cluster.start_sample < ignore_spikes_threshold) {
            // discard spurious clusters
            continue;
        }

        if (merged_clusters.empty()) {
            merged_clusters.push_back(std::move(cluster));
            continue;
        }

        auto& last_cluster = merged_clusters.back();
        if (cluster.start_sample - last_cluster.end_sample > cluster_dist) {
            // new cluster is too far away to merge, just accept it
            merged_clusters.push_back(std::move(cluster));
        } else {
            // extend previous cluster and update metadata
            last_cluster.end_sample = cluster.end_sample;
            if (cluster.max_val >= last_cluster.max_val) {
                last_cluster.max_val = cluster.max_val;
                last_cluster.argmax_sample = cluster.argmax_sample;
            }
        }
    }

    return merged_clusters;
}
template SampleRanges<c10::Half> detect_pore_signal(const at::Tensor& signal,
                                                    c10::Half threshold,
                                                    uint64_t cluster_dist,
                                                    uint64_t ignore_prefix,
                                                    uint64_t ignore_spikes_threshold);
template SampleRanges<float> detect_pore_signal(const at::Tensor& signal,
                                                float threshold,
                                                uint64_t cluster_dist,
                                                uint64_t ignore_prefix,
                                                uint64_t ignore_spikes_threshold);
template SampleRanges<int16_t> detect_pore_signal(const at::Tensor& signal,
                                                  int16_t threshold,
                                                  uint64_t cluster_dist,
                                                  uint64_t ignore_prefix,
                                                  uint64_t ignore_spikes_threshold);

}  // namespace dorado::splitter
