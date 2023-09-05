#include "PolyACalculator.h"

#include "3rdparty/edlib/edlib/include/edlib.h"
#include "utils/sequence_utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace {

int kMaxTailLength = 500;

std::pair<int, int> determine_signal_bounds5(int signal_anchor,
                                             const c10::Half* signal,
                                             const std::vector<uint64_t>& seq_to_sig_map,
                                             bool fwd,
                                             const dorado::ReadPtr& read,
                                             int num_samples_per_base) {
    int signal_len = seq_to_sig_map[seq_to_sig_map.size() - 1];

    // Maximum gap between intervals that can be combined.
    const int kMaxSampleGap = num_samples_per_base * 3;

    auto check_for_peak = [&](float ref_x, float thres, int start, int end) -> bool {
        float max = signal[start];
        for (int i = start + 1; i < end; i++) {
            max = std::max(max, float(signal[i]));
        }
        return (std::abs(ref_x - max) > thres);
    };

    auto calc_stats = [&](int s, int e) -> std::pair<float, float> {
        float avg = 0;
        for (int i = s; i < e; i++) {
            avg += signal[i];
        }
        avg = avg / (e - s);
        float var = 0;
        for (int i = s; i < e; i++) {
            var += (signal[i] - avg) * (signal[i] - avg);
        }
        var = var / (e - s);
        return {avg, std::sqrt(var)};
    };

    std::vector<std::pair<int, int>> intervals;
    std::vector<std::pair<float, float>> interval_stats;
    std::pair<int, int> last_interval{0, 0};

    // Maximum variance between consecutive values to be
    // considered part of the same interval.
    const float kVar = 0.35f;

    // Determine the outer boundary of the signal space to
    // consider based on the anchor. Go further in the direction
    // the polyA/T tail is running, and less in the other, assuming
    // that the anchor provides a reasonable starting position.
    const int kSpread = num_samples_per_base * kMaxTailLength;
    int left_end = (fwd ? std::max(0, signal_anchor - kSpread)
                        : std::max(0, signal_anchor - kSpread / 10));
    int right_end = (fwd ? std::min(signal_len, signal_anchor + kSpread / 10)
                         : std::min(signal_len, signal_anchor + kSpread));

    const int kNum = 25;
    spdlog::debug("Bounds left {}, right {}", left_end, right_end);
    for (int i = left_end; i < right_end; i += kNum) {
        int s = i, e = std::min(i + kNum, right_end);
        auto [avg, stdev] = calc_stats(s, e);
        if (stdev < kVar) {
            if (intervals.size() > 1 && intervals.back().second == i &&
                std::abs(avg - interval_stats.back().first) < 0.2 &&
                std::abs(stdev - interval_stats.back().second) < 0.1) {
                auto& last = intervals.back();
                last.second = e;
                spdlog::debug("update interval {}-{}, stdev {}", last.first, last.second, stdev);
            } else {
                intervals.push_back({i, i + kNum});
                spdlog::debug("new interval {}-{}, stdev {}", s, e, stdev);
            }
            interval_stats.push_back({avg, stdev});
        }
    }

    //std::vector<std::pair<int, int>> merged_intervals;
    //for(auto intvl : intervals) {
    //    if (merged_intervals.empty()) {
    //        merged_intervals.push_bacl(intvl);
    //    } else {
    //        auto& prev = merged_intervals.back();
    //        if (intvl.first - prev.second < kMaxSampleGap) {
    //            prev.second = intvl.second;
    //        } else {
    //            merged_intervals.push_back(intvl);
    //        }
    //    }
    //}

    std::vector<std::pair<int, int>> filtered_intervals;
    // In forward strand, the poly A/T signal should end within 100bp of the
    // signal anchor, and in reverse strand it should start within 100bp of the
    // anchor.
    int kAnchorProximity = 100 * num_samples_per_base;
    if (fwd) {
        std::copy_if(
                intervals.begin(), intervals.end(), std::back_inserter(filtered_intervals),
                [&](auto& i) { return std::abs(signal_anchor - i.second) < kAnchorProximity; });
    } else {
        std::copy_if(intervals.begin(), intervals.end(), std::back_inserter(filtered_intervals),
                     [&](auto& i) { return std::abs(signal_anchor - i.first) < kAnchorProximity; });
    }

    if (filtered_intervals.empty()) {
        spdlog::debug("Anchor {} No range within anchor proximity found", signal_anchor);
        return {0, 0};
    }

    auto best_interval = std::max_element(
            filtered_intervals.begin(), filtered_intervals.end(),
            [](auto& l, auto& r) { return (l.second - l.first) < (r.second - r.first); });

    spdlog::debug("Anchor {} Range {} {}", signal_anchor, best_interval->first,
                  best_interval->second);
    return *best_interval;
}

std::pair<int, int> determine_signal_bounds3(int signal_end,
                                             const c10::Half* signal,
                                             const std::vector<uint64_t>& seq_to_sig_map,
                                             bool fwd,
                                             const std::shared_ptr<dorado::Read>& read) {
    const int kNum = 50;
    std::array<float, kNum> inputs;
    auto stats = [&inputs]() -> std::pair<float, float> {
        float avg = 0;
        for (auto x : inputs) {
            avg += x;
        }
        avg = avg / inputs.size();
        float var = 0;
        for (auto x : inputs) {
            var += (x - avg) * (x - avg);
        }
        var = var / inputs.size();
        return {avg, std::sqrt(var)};
    };
    auto smoother = [&inputs](int n, float x) -> float {
        const float factor = 0.5;
        float val = 0;
        for (int i = 0; i < inputs.size(); i++) {
            val += inputs[i];
        }
        val /= inputs.size();
        return factor * val + (1 - factor) * x;
    };
    int signal_start = 0;
    int n = 0;
    for (int i = signal_end; (fwd ? i > 0 : i < read->raw_data.size(0)); (fwd ? i-- : i++)) {
        float raw_x = signal[i];
        float x = smoother(n, raw_x);
        auto [avg, stdev] = stats();
        spdlog::debug("idx {} x {}, avg {}, stdev {}", i, x, avg, stdev);
        if (n > kNum && std::abs(x - avg) > 2 * stdev) {
            spdlog::debug("Reached end at {} at mean {} stdev {}", i, avg, stdev);
            break;
        }
        inputs[n % inputs.size()] = raw_x;
        signal_start = i;
        n++;
    }
    if (!fwd) {
        std::swap(signal_start, signal_end);
    }
    return {signal_start, signal_end};
}

int estimate_samples_per_base(const dorado::ReadPtr& read) {
    float num_samples_per_base = static_cast<float>(read->raw_data.size(0)) / read->seq.length();
    return static_cast<int>(num_samples_per_base);
}

constexpr int count_trailing_chars(const std::string_view adapter, char c) {
    int count = 0;
    for (int i = adapter.length() - 1; i >= 0; i--) {
        if (adapter[i] == c) {
            count++;
        } else {
            break;
        }
    }
    return count;
}
}  // namespace

namespace dorado {

void PolyACalculator::worker_thread() {
    torch::InferenceMode inference_mode_guard;

    Message message;
    while (get_input_message(message)) {
        // If this message isn't a read, just forward it to the sink.
        if (!std::holds_alternative<ReadPtr>(message)) {
            send_message_to_sink(std::move(message));
            continue;
        }

        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto read = std::get<ReadPtr>(std::move(message));

        const std::string SSP = "TTTCTGTTGGTGCTGATATTGCTTT";
        const std::string SSP_rc = utils::reverse_complement(SSP);
        const std::string VNP = "ACTTGCCTGTCGCTCTATCTTCAGAGGAGAGTCCGCCGCCCGCAAGTTTT";
        const std::string VNP_rc = utils::reverse_complement(VNP);
        int trailing_Ts = count_trailing_chars(VNP, 'T');

        int window_size = 150;
        std::string read_top = read->seq.substr(0, window_size);
        auto bottom_start = std::max(0, (int)read->seq.length() - window_size);
        std::string read_bottom = read->seq.substr(bottom_start, window_size);

        EdlibAlignConfig align_config = edlibDefaultAlignConfig();
        align_config.task = EDLIB_TASK_LOC;
        align_config.mode = EDLIB_MODE_HW;

        // Check for forward strand.
        EdlibAlignResult top_v1 = edlibAlign(SSP.data(), SSP.length(), read_top.data(),
                                             read_top.length(), align_config);
        EdlibAlignResult bottom_v1 = edlibAlign(VNP_rc.data(), VNP_rc.length(), read_bottom.data(),
                                                read_bottom.length(), align_config);

        int dist_v1 = top_v1.editDistance + bottom_v1.editDistance;

        // Check for reverse strand.
        EdlibAlignResult top_v2 = edlibAlign(VNP.data(), VNP.length(), read_top.data(),
                                             read_top.length(), align_config);
        EdlibAlignResult bottom_v2 = edlibAlign(SSP_rc.data(), SSP_rc.length(), read_bottom.data(),
                                                read_bottom.length(), align_config);

        int dist_v2 = top_v2.editDistance + bottom_v2.editDistance;
        spdlog::debug("v1 dist {}, v2 dist {}", dist_v1, dist_v2);

        bool proceed = std::min(dist_v1, dist_v2) < 30;

        if (proceed) {
            bool fwd = true;
            int start = 0, end = 0;
            if (dist_v2 < dist_v1) {
                fwd = false;
                start = top_v2.endLocations[0];
            } else {
                end = bottom_start + bottom_v1.startLocations[0];
            }

            const auto stride = read->model_stride;
            const auto seq_to_sig_map = utils::moves_to_map(
                    read->moves, stride, read->raw_data.size(0), read->seq.size() + 1);

            int signal_anchor = fwd ? seq_to_sig_map[end] : seq_to_sig_map[start];
            spdlog::debug(
                    "Strand {}; poly A/T signal anchor {}, shift/scale {} {}, samples trimmed {}",
                    fwd ? '+' : '-', signal_anchor, read->mshift, read->mscale,
                    read->num_trimmed_samples);

            const c10::Half* signal = static_cast<c10::Half*>(read->raw_data.data_ptr());

            auto num_samples_per_base = estimate_samples_per_base(read);

            // Walk through signal
            auto [signal_start, signal_end] = determine_signal_bounds5(
                    signal_anchor, signal, seq_to_sig_map, fwd, read, num_samples_per_base);

            int num_bases = static_cast<int>((signal_end - signal_start) / num_samples_per_base) -
                            trailing_Ts;
            if (num_bases >= 0 && num_bases < kMaxTailLength) {
                spdlog::debug(
                        "{} PolyA bases {}, signal anchor {} region {} Signal range is {} {}, "
                        "primer {}, "
                        "samples/base {}",
                        read->read_id, num_bases, signal_anchor,
                        read->seq.substr(fwd ? std::max(0, (end - num_bases)) : start, num_bases),
                        signal_start, signal_end, fwd ? end : start, num_samples_per_base);
                polyA += num_bases;
                read->rna_poly_tail_length = num_bases;
            } else {
                spdlog::warn(
                        "{} PolyA bases {}, signal anchor {} Signal range is {} {} primer {}, "
                        "samples/base {}",
                        read->read_id, num_bases, signal_anchor, signal_start, signal_end,
                        fwd ? end : start, num_samples_per_base);
                not_called++;
            }
        } else {
            spdlog::warn("{} primer edit distance too high {}", read->read_id,
                         std::min(dist_v1, dist_v2));
            not_called++;
        }

        num_reads += 1;
        send_message_to_sink(std::move(read));
    }
}

PolyACalculator::PolyACalculator(size_t num_worker_threads, size_t max_reads)
        : MessageSink(max_reads), m_num_worker_threads(num_worker_threads) {
    start_threads();
}

void PolyACalculator::start_threads() {
    for (size_t i = 0; i < m_num_worker_threads; i++) {
        m_workers.push_back(
                std::make_unique<std::thread>(std::thread(&PolyACalculator::worker_thread, this)));
    }
}

void PolyACalculator::terminate_impl() {
    terminate_input_queue();
    for (auto& m : m_workers) {
        if (m->joinable()) {
            m->join();
        }
    }
    m_workers.clear();
    spdlog::info("Total {}, not called {}, Avg polyA length {}", num_reads.load(),
                 not_called.load(), polyA.load() / num_reads.load());
}

void PolyACalculator::restart() {
    restart_input_queue();
    start_threads();
}

}  // namespace dorado
