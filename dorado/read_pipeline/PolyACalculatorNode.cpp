#include "PolyACalculatorNode.h"

#include "utils/math_utils.h"
#include "utils/sequence_utils.h"

#include <edlib.h>
#include <spdlog/spdlog.h>
#include <toml.hpp>
#include <toml/value.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

namespace {

using PolyAConfig = dorado::PolyACalculatorNode::PolyAConfig;

// Prepare the PolyA configuration struct. If a configuration
// file is available, parse it to extract parameters. Otherwise
// prepare the default configuration.
PolyAConfig prepare_config(const std::string* config_file) {
    PolyAConfig config;

    if (config_file != nullptr) {
        const toml::value config_toml = toml::parse(*config_file);

        if (config_toml.contains("anchors")) {
            const auto& anchors = toml::find(config_toml, "anchors");

            if (anchors.contains("front_primer") || anchors.contains("rear_primer")) {
                if (!(anchors.contains("front_primer") && anchors.contains("rear_primer"))) {
                    throw std::runtime_error(
                            "Both front_primer and rear_primer must be provided in the PolyA "
                            "configuration file.");
                }
                config.front_primer = toml::find<std::string>(anchors, "front_primer");
                config.rear_primer = toml::find<std::string>(anchors, "rear_primer");
            }

            if (anchors.contains("plasmid_front_flank") || anchors.contains("plasmid_rear_flank")) {
                if (!(anchors.contains("plasmid_front_flank") &&
                      anchors.contains("plasmid_rear_flank"))) {
                    throw std::runtime_error(
                            "Both plasmid_front_flank and plasmid_rear_flank must be provided in "
                            "the PolyA configuration file.");
                }
                config.plasmid_front_flank =
                        toml::find<std::string>(anchors, "plasmid_front_flank");
                config.plasmid_rear_flank = toml::find<std::string>(anchors, "plasmid_rear_flank");
                config.is_plasmid = true;
            }
        }

        if (config_toml.contains("tail")) {
            const auto& tail = toml::find(config_toml, "tail");

            if (tail.contains("tail_interrupt_length")) {
                config.tail_interrupt_length = toml::find<int>(tail, "tail_interrupt_length");
            }
        }
    }

    if (!config.front_primer.empty()) {
        config.rc_front_primer = dorado::utils::reverse_complement(config.front_primer);
    }
    if (!config.rear_primer.empty()) {
        config.rc_rear_primer = dorado::utils::reverse_complement(config.rear_primer);
    }
    if (!config.plasmid_front_flank.empty()) {
        config.rc_plasmid_front_flank =
                dorado::utils::reverse_complement(config.plasmid_front_flank);
        spdlog::info("{} {}", config.plasmid_rear_flank, config.rc_plasmid_rear_flank);
    }
    if (!config.plasmid_rear_flank.empty()) {
        config.rc_plasmid_rear_flank = dorado::utils::reverse_complement(config.plasmid_rear_flank);
    }

    return config;
}

struct SignalAnchorInfo {
    // Is the strand in forward or reverse direction.
    bool is_fwd_strand = true;
    // The start or end anchor for the polyA/T signal
    // depending on whether the strand is forward or
    // reverse.
    int signal_anchor = -1;
    // Number of additional A/T bases in the polyA
    // stretch from the adapter.
    int trailing_adapter_bases = 0;
};

const int kMaxTailLength = 750;

// This algorithm walks through the signal in windows. For each window
// the avg and stdev of the signal is computed. If the stdev is below
// an empirically determined threshold, and consecutive windows have
// similar avg and stdev, then those windows are considered to be part
// of the polyA tail.
std::pair<int, int> determine_signal_bounds(
        int signal_anchor,
        bool fwd,
        const dorado::SimplexRead& read,
        float num_samples_per_base,
        bool is_rna,
        const dorado::PolyACalculatorNode::PolyAConfig& config) {
    const c10::Half* signal = static_cast<c10::Half*>(read.read_common.raw_data.data_ptr());
    int signal_len = int(read.read_common.get_raw_data_samples());

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
    std::pair<float, float> last_interval_stats;

    // Maximum variance between consecutive values to be
    // considered part of the same interval.
    const float kVar = 0.35f;
    // Determine the outer boundary of the signal space to
    // consider based on the anchor.
    const int kSpread = int(std::round(num_samples_per_base * kMaxTailLength));
    // Maximum gap between intervals that can be combined.
    const int kMaxSampleGap = int(std::round(num_samples_per_base * 5));
    // Minimum size of intervals considered for merge.
    const int kMinIntervalSizeForMerge =
            std::max(static_cast<int>(std::round(10 * num_samples_per_base)), 200);
    // Floor for average signal value of poly tail.
    const float kMinAvgVal = (is_rna ? 0.0f : -3.0f);

    int left_end = is_rna ? std::max(0, signal_anchor - 50) : std::max(0, signal_anchor - kSpread);
    int right_end = std::min(signal_len, signal_anchor + kSpread);
    spdlog::trace("Bounds left {}, right {}", left_end, right_end);

    const int kStride = 3;
    for (int s = left_end; s < right_end; s += kStride) {
        int e = std::min(s + kMaxSampleGap, right_end);
        auto [avg, stdev] = calc_stats(s, e);
        if (stdev < kVar) {
            // If a new interval overlaps with the previous interval, just extend
            // the previous interval.
            if (intervals.size() > 1 && intervals.back().second >= s &&
                std::abs(avg - last_interval_stats.first) < 0.2 && (avg > kMinAvgVal)) {
                auto& last = intervals.back();
                spdlog::trace("extend interval {}-{} to {}-{} avg {} stdev {}", last.first,
                              last.second, s, e, avg, stdev);
                last.second = e;
            } else {
                // Attempt to merge the most recent interval and the one before
                // that if the gap between the intervals is small and both of the
                // intervals are longer than some threshold.
                if (intervals.size() >= 2) {
                    auto& last = intervals.back();
                    auto& second_last = intervals[intervals.size() - 2];
                    spdlog::trace("Evaluate for merge {}-{} with {}-{}", second_last.first,
                                  second_last.second, last.first, last.second);
                    if ((last.first - second_last.second < kMaxSampleGap) &&
                        (last.second - last.first > kMinIntervalSizeForMerge) &&
                        (second_last.second - second_last.first > kMinIntervalSizeForMerge)) {
                        spdlog::trace("Merge interval {}-{} with {}-{}", second_last.first,
                                      second_last.second, second_last.first, last.second);
                        second_last.second = last.second;
                        intervals.pop_back();
                    } else if (second_last.second - second_last.first <
                               std::round(num_samples_per_base * config.min_base_count)) {
                        intervals.erase(intervals.end() - 2);
                    }
                }
                spdlog::trace("Add new interval {}-{} avg {} stdev {}", s, e, avg, stdev);
                intervals.push_back({s, e});
            }
            last_interval_stats = {avg, stdev};
        }
    }

    std::string int_str = "";
    for (const auto& in : intervals) {
        int_str += std::to_string(in.first) + "-" + std::to_string(in.second) + ", ";
    }
    spdlog::trace("found intervals {}", int_str);

    // Cluster intervals
    const int kMaxInterruption =
            int(std::round(num_samples_per_base * config.tail_interrupt_length));
    std::vector<std::pair<int, int>> clustered_intervals;
    for (const auto& i : intervals) {
        if (clustered_intervals.empty()) {
            clustered_intervals.push_back(i);
        } else {
            auto& last = clustered_intervals.back();
            if (std::abs(i.first - last.second) < kMaxInterruption) {
                last.second = i.second;
            } else {
                clustered_intervals.push_back(i);
            }
        }
    }

    int_str = "";
    for (auto in : clustered_intervals) {
        int_str += std::to_string(in.first) + "-" + std::to_string(in.second) + ", ";
    }
    spdlog::trace("clustered intervals {}", int_str);

    std::vector<std::pair<int, int>> filtered_intervals;
    std::copy_if(
            clustered_intervals.begin(), clustered_intervals.end(),
            std::back_inserter(filtered_intervals), [&](auto& i) {
                int interval_size = i.second - i.first;
                // Filter out any small intervals.
                if (interval_size < (std::round(num_samples_per_base * config.min_base_count))) {
                    return false;
                }
                // Only keep intervals that are close-ish to the signal anchor or to the previous interval.
                bool within_anchor_dist =
                        (std::abs(signal_anchor - i.second) < interval_size ||
                         std::abs(signal_anchor - i.first) < interval_size ||
                         ((i.first <= signal_anchor) && (signal_anchor <= i.second)));

                return within_anchor_dist;
            });

    int_str = "";
    for (auto in : filtered_intervals) {
        int_str += std::to_string(in.first) + "-" + std::to_string(in.second) + ", ";
    }
    spdlog::trace("filtered intervals {}", int_str);

    if (filtered_intervals.empty()) {
        spdlog::trace("Anchor {} No range within anchor proximity found", signal_anchor);
        return {0, 0};
    }

    // Choose the longest interval. If there is a tie for the longest interval,
    // choose the one that is closest to the anchor.
    auto best_interval = std::max_element(filtered_intervals.begin(), filtered_intervals.end(),
                                          [&](auto& l, auto& r) {
                                              auto l_size = l.second - l.first;
                                              auto r_size = r.second - r.first;
                                              if (l_size != r_size) {
                                                  return l_size < r_size;
                                              } else {
                                                  if (fwd) {
                                                      return std::abs(l.second - signal_anchor) <
                                                             std::abs(r.second - signal_anchor);
                                                  } else {
                                                      return std::abs(l.first - signal_anchor) <
                                                             std::abs(r.first - signal_anchor);
                                                  }
                                              }
                                          });

    spdlog::trace("Anchor {} Range {} {}", signal_anchor, best_interval->first,
                  best_interval->second);

    return *best_interval;
}

// Estimate the number of samples per base.
// For RNA, use the mean of 10-90th percentile samples/base estimated from
// the move table.
// For DNA, just take the median samples/base estimated from the move table.
float estimate_samples_per_base(const dorado::SimplexRead& read, bool is_rna) {
    const size_t num_bases = read.read_common.seq.length();
    const auto num_samples = read.read_common.get_raw_data_samples();
    const auto stride = read.read_common.model_stride;
    const auto seq_to_sig_map =
            dorado::utils::moves_to_map(read.read_common.moves, stride, num_samples, num_bases + 1);
    // Store the samples per base in float to use the quantile calcuation function.
    std::vector<float> sizes(seq_to_sig_map.size() - 1, 0.f);
    for (int i = 1; i < int(seq_to_sig_map.size()); i++) {
        sizes[i - 1] = static_cast<float>(seq_to_sig_map[i] - seq_to_sig_map[i - 1]);
    }

    if (is_rna) {
        auto quantiles = dorado::utils::quantiles(sizes, {0.1f, 0.9f});
        float sum = 0.f;
        int count = 0;
        for (auto s : sizes) {
            if (s >= quantiles[0] && s <= quantiles[1]) {
                sum += s;
                count++;
            }
        }
        return (count > 0 ? (sum / count) : 0.f);
    } else {
        auto quantiles = dorado::utils::quantiles(sizes, {0.5});
        return static_cast<float>(quantiles[0]);
    }
}

// In order to find the approximate location of the start/end (anchor) of the polyA
// cDNA tail, the adapter ends are aligned to the reads to find the breakpoint
// between the read and the adapter. Adapter alignment also helps determine
// the strand direction. This function returns a struct with the strand direction,
// the approximate anchor for the tail, and if there needs to be an adjustment
// made to the final polyA tail count based on the adapter sequence (e.g. because
// the adapter itself contains several As).
SignalAnchorInfo determine_signal_anchor_and_strand_cdna(const dorado::SimplexRead& read,
                                                         const PolyAConfig& config) {
    const std::string& front_primer = config.front_primer;
    const std::string& front_primer_rc = config.rc_front_primer;
    const std::string& rear_primer = config.rear_primer;
    const std::string& rear_primer_rc = config.rc_rear_primer;
    int trailing_Ts = dorado::utils::count_trailing_chars(rear_primer, 'T');

    const int kWindowSize = 150;
    std::string read_top = read.read_common.seq.substr(0, kWindowSize);
    auto bottom_start = std::max(0, (int)read.read_common.seq.length() - kWindowSize);
    std::string read_bottom = read.read_common.seq.substr(bottom_start, kWindowSize);

    EdlibAlignConfig align_config = edlibDefaultAlignConfig();
    align_config.task = EDLIB_TASK_LOC;
    align_config.mode = EDLIB_MODE_HW;

    // Check for forward strand.
    EdlibAlignResult top_v1 = edlibAlign(front_primer.data(), int(front_primer.length()),
                                         read_top.data(), int(read_top.length()), align_config);
    EdlibAlignResult bottom_v1 =
            edlibAlign(rear_primer_rc.data(), int(rear_primer_rc.length()), read_bottom.data(),
                       int(read_bottom.length()), align_config);

    int dist_v1 = top_v1.editDistance + bottom_v1.editDistance;

    // Check for reverse strand.
    EdlibAlignResult top_v2 = edlibAlign(rear_primer.data(), int(rear_primer.length()),
                                         read_top.data(), int(read_top.length()), align_config);
    EdlibAlignResult bottom_v2 =
            edlibAlign(front_primer_rc.data(), int(front_primer_rc.length()), read_bottom.data(),
                       int(read_bottom.length()), align_config);

    int dist_v2 = top_v2.editDistance + bottom_v2.editDistance;
    spdlog::trace("v1 dist {}, v2 dist {}", dist_v1, dist_v2);

    bool fwd = dist_v1 < dist_v2;
    bool proceed = std::min(dist_v1, dist_v2) < 30 && std::abs(dist_v1 - dist_v2) > 10;

    SignalAnchorInfo result = {false, -1, trailing_Ts};

    if (proceed) {
        int base_anchor = 0;
        if (fwd) {
            base_anchor = bottom_start + bottom_v1.startLocations[0];
        } else {
            base_anchor = top_v2.endLocations[0];
        }

        const auto stride = read.read_common.model_stride;
        const auto seq_to_sig_map = dorado::utils::moves_to_map(
                read.read_common.moves, stride, read.read_common.get_raw_data_samples(),
                read.read_common.seq.size() + 1);
        int signal_anchor = int(seq_to_sig_map[base_anchor]);

        result = {fwd, signal_anchor, trailing_Ts};
    } else {
        spdlog::debug("{} primer edit distance too high {}", read.read_common.read_id,
                      std::min(dist_v1, dist_v2));
    }

    edlibFreeAlignResult(top_v1);
    edlibFreeAlignResult(bottom_v1);
    edlibFreeAlignResult(top_v2);
    edlibFreeAlignResult(bottom_v2);

    return result;
}

// RNA polyA appears at the beginning of the strand. Since the adapter
// for RNA has been trimmed off already, the polyA search can begin
// from the start of the signal.
SignalAnchorInfo determine_signal_anchor_and_strand_drna(const dorado::SimplexRead& read) {
    return SignalAnchorInfo{false, read.read_common.rna_adapter_end_signal_pos, 0};
}

}  // namespace

namespace dorado {

void PolyACalculatorNode::input_thread_fn() {
    at::InferenceMode inference_mode_guard;

    Message message;
    while (get_input_message(message)) {
        // If this message isn't a read, just forward it to the sink.
        if (!std::holds_alternative<SimplexReadPtr>(message)) {
            send_message_to_sink(std::move(message));
            continue;
        }

        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto read = std::get<SimplexReadPtr>(std::move(message));

        // Determine the strand direction, approximate base space anchor for the tail, and whether
        // the final length needs to be adjusted depending on the adapter sequence.
        auto [fwd, signal_anchor, trailing_Ts] =
                m_is_rna ? determine_signal_anchor_and_strand_drna(*read)
                         : determine_signal_anchor_and_strand_cdna(*read, m_config);

        if (signal_anchor >= 0) {
            spdlog::debug("{} Strand {}; poly A/T signal anchor {}", read->read_common.read_id,
                          fwd ? '+' : '-', signal_anchor);

            auto num_samples_per_base = estimate_samples_per_base(*read, m_is_rna);

            // Walk through signal. Require a minimum of length 10 poly-A since below that
            // the current algorithm returns a lot of false intervals.
            auto [signal_start, signal_end] = determine_signal_bounds(
                    signal_anchor, fwd, *read, num_samples_per_base, m_is_rna, m_config);
            auto signal_len = signal_end - signal_start;

            // Create an offset for dRNA data. There is a tendency to overestimate the length of dRNA
            // tails, especially shorter ones. This correction factor appears to fix the bias
            // for most dRNA data. This exponential fit was done based on the standards data.
            // TODO: In order to improve this, perhaps another pass over the tail interval is needed
            // to get a more refined boundary estimation?
            if (m_is_rna) {
                signal_len -= int(std::round(std::min(
                        100.f, std::exp(5.6838f - 0.0021f * static_cast<float>(signal_len)))));
            }

            int num_bases = int(std::round(static_cast<float>(signal_len) / num_samples_per_base)) -
                            trailing_Ts;

            if (num_bases > 0 && num_bases < kMaxTailLength) {
                spdlog::debug(
                        "{} PolyA bases {}, signal anchor {} Signal range is {} {} Signal length "
                        "{}, "
                        "samples/base {} trim {} read len {}",
                        read->read_common.read_id, num_bases, signal_anchor, signal_start,
                        signal_end, signal_len, num_samples_per_base,
                        read->read_common.num_trimmed_samples, read->read_common.seq.length());

                // Set tail length property in the read.
                read->read_common.rna_poly_tail_length = num_bases;

                // Update debug stats.
                total_tail_lengths_called += num_bases;
                ++num_called;
                if (spdlog::get_level() <= spdlog::level::debug) {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    tail_length_counts[num_bases]++;
                }
            } else {
                spdlog::debug(
                        "{} PolyA bases {}, signal anchor {} Signal range is {}, "
                        "samples/base {}, trim  {}",
                        read->read_common.read_id, num_bases, signal_anchor, signal_start,
                        signal_end, num_samples_per_base, read->read_common.num_trimmed_samples);
                num_not_called++;
            }
        } else {
            num_not_called++;
        }

        send_message_to_sink(std::move(read));
    }
}

PolyACalculatorNode::PolyACalculatorNode(size_t num_worker_threads,
                                         bool is_rna,
                                         size_t max_reads,
                                         const std::string* config_file)
        : MessageSink(max_reads, static_cast<int>(num_worker_threads)), m_is_rna(is_rna) {
    m_config = prepare_config(config_file);
    start_input_processing(&PolyACalculatorNode::input_thread_fn, this);
}

void PolyACalculatorNode::terminate_impl() {
    stop_input_processing();

    spdlog::debug("Total called {}, not called {}, avg tail length {}", num_called.load(),
                  num_not_called.load(),
                  num_called.load() > 0 ? total_tail_lengths_called.load() / num_called.load() : 0);

    // Visualize a distribution of the tail lengths called.
    static bool done = false;
    if (!done && (spdlog::get_level() <= spdlog::level::debug)) {
        int max_val = -1;
        for (auto [k, v] : tail_length_counts) {
            max_val = std::max(v, max_val);
        }
        int factor = std::max(1, 1 + max_val / 100);
        for (auto [k, v] : tail_length_counts) {
            spdlog::debug("{:03d} : {}", k, std::string(v / factor, '*'));
        }
        done = true;
    }
}

stats::NamedStats PolyACalculatorNode::sample_stats() const {
    stats::NamedStats stats = stats::from_obj(m_work_queue);
    stats["reads_not_estimated"] = static_cast<double>(num_not_called.load());
    stats["reads_estimated"] = static_cast<double>(num_called.load());
    stats["average_tail_length"] = static_cast<double>(
            num_called.load() > 0 ? total_tail_lengths_called.load() / num_called.load() : 0);
    return stats;
}

}  // namespace dorado
