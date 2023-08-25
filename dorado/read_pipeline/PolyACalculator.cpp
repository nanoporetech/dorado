#include "PolyACalculator.h"

#include "3rdparty/edlib/edlib/include/edlib.h"
#include "utils/sequence_utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace {

int kMaxTailLength = 500;

std::pair<int, int> determine_signal_bounds4(int signal_anchor,
                                             const c10::Half* signal,
                                             const std::vector<uint64_t>& seq_to_sig_map,
                                             bool fwd,
                                             const dorado::ReadPtr& read,
                                             int num_samples_per_base) {
    int signal_len = seq_to_sig_map[seq_to_sig_map.size() - 1];

    // Maximum gap between intervals that can be combined.
    const int kMaxSampleGap = num_samples_per_base * 3;

    // Helper function to returned smoothed values from
    // the signal.
    auto smoother = [&](int i) -> float {
        if (i == 0) {
            return signal[i];
        }
        int s = std::max(0, i - 3);
        int e = i;
        float val = 0.f;
        for (int i = s; i < e; i++) {
            val += signal[i];
        }
        return val / (e - s);
    };

    auto check_for_peak = [&](float ref_x, float thres, int start, int end) -> bool {
        float max = signal[start];
        for (int i = start + 1; i < end; i++) {
            max = std::max(max, float(signal[i]));
        }
        return (std::abs(ref_x - max) > thres);
    };

    std::vector<std::pair<int, int>> intervals;
    std::pair<int, int> last_interval{0, 0};
    int last_interval_len = 0;
    float last_x_last_interval = 0.f;

    // Maximum variance between consecutive values to be
    // considered part of the same interval.
    const float kVar = 0.4f;

    // Determine the outer boundary of the signal space to
    // consider based on the anchor. Go further in the direction
    // the polyA/T tail is running, and less in the other, assuming
    // that the anchor provides a reasonable starting position.
    const int kSpread = num_samples_per_base * kMaxTailLength;
    int left_end = (fwd ? std::max(0, signal_anchor - kSpread)
                        : std::max(0, signal_anchor - kSpread / 10));
    int right_end = (fwd ? std::min(signal_len, signal_anchor + kSpread / 10)
                         : std::min(signal_len, signal_anchor + kSpread));

    float x_last = smoother(0);  // Compare new values against an average of last few values.
    int signal_start = left_end, signal_end = signal_start;
    for (int i = left_end; i < right_end; i++) {
        float x = signal[i];
        if (std::abs(x - x_last) < kVar) {
            signal_end = i;
        } else {
            int range = signal_end - signal_start;
            if (range > (num_samples_per_base * 3)) {
                // Opportunistically merge consecutive intervals if they look like
                // they could belong to the same run.
                if (signal_start - last_interval.second < kMaxSampleGap &&
                    std::abs(x_last - last_x_last_interval) < kVar &&
                    check_for_peak(x_last, kVar, last_interval.second, signal_start)) {
                    signal_start = last_interval.first;
                    last_interval = {signal_start, signal_end};
                    last_interval_len = signal_end - signal_start;
                    spdlog::debug("Update range {} {}", last_interval.first, last_interval.second);
                    auto last_ref = intervals.back();
                    last_ref = last_interval;
                } else if (range > last_interval_len) {
                    last_interval = {signal_start, signal_end};
                    last_interval_len = signal_end - signal_start;
                    spdlog::debug("New range {} {}", last_interval.first, last_interval.second);
                    last_x_last_interval = x_last;
                    intervals.push_back(last_interval);
                }
            }
            signal_start = i;
        }
        x_last = smoother(i);
    }

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

std::pair<int, int> determine_signal_bounds(int signal_end,
                                            const c10::Half* signal,
                                            const std::vector<uint64_t>& seq_to_sig_map,
                                            bool fwd,
                                            const dorado::ReadPtr& read) {
    std::array<float, 25> inputs;
    auto smoother = [&inputs](float x) -> float {
        const float factor = 0.4;
        float val = 0;
        for (int i = 0; i < inputs.size(); i++) {
            val += inputs[i];
        }
        val /= inputs.size();
        return factor * val + (1 - factor) * x;
    };
    float x_n = 0, x_n_1 = 0;
    float v_n = 1, v_n_1 = 0;
    int n = 0;
    float stdev;
    int signal_start = 0;
    for (int i = signal_end; (fwd ? i > 0 : i < read->raw_data.size(0)); (fwd ? i-- : i++)) {
        float x = signal[i];
        float smooth_x = smoother(x);
        x_n_1 = x_n;
        v_n_1 = v_n;
        if (n == 25) {
            spdlog::debug("idx {} input {}, Mean {} stddev {}", i, x, x_n, stdev);
        }
        if (n > 25 and std::abs(smooth_x - x_n_1) > 2 * stdev) {
            spdlog::debug("Reached end at {} for x {} (raw {})  at mean {} stdev {}", i, smooth_x,
                          x, x_n, stdev);
            break;
        }
        inputs[n % inputs.size()] = x;
        n++;
        x_n = x_n_1 + float(x - x_n_1) / (n + 1);
        if (n < 30) {
            v_n = v_n_1 + float((x - x_n_1) * (x - x_n) - v_n_1) / (n + 1);
            stdev = std::sqrt(v_n);
        }
        signal_start = i;
    }
    spdlog::debug("Loop end at mean {} stdev {}", x_n, stdev);
    if (!fwd) {
        std::swap(signal_start, signal_end);
    }
    const int kSignalCorrection = 15;  // Approximate overshoot of signal detection algorithm.
    return {signal_start, signal_end - kSignalCorrection};
}

std::pair<int, int> determine_signal_bounds2(int signal_end,
                                             const c10::Half* signal,
                                             const std::vector<uint64_t>& seq_to_sig_map,
                                             bool fwd,
                                             const std::shared_ptr<dorado::Read>& read) {
    // Determine avg signal val for A or T.
    float x_n = 0, x_n_1 = 0;
    float v_n = 1, v_n_1 = 0;
    int n = 0;
    float stdev;
    int signal_start;
    for (int i = 0; i < read->seq.length(); i++) {
        if (read->seq[i] == (fwd ? 'A' : 'T')) {
            auto s = seq_to_sig_map[i];
            auto e = seq_to_sig_map[i + 1];
            for (int j = s; j < e; j++) {
                n++;
                x_n_1 = x_n;
                v_n_1 = v_n;
                float x = signal[j];
                x_n = x_n_1 + float(x - x_n_1) / (n + 1);
                v_n = v_n_1 + float((x - x_n_1) * (x - x_n) - v_n_1) / (n + 1);
                stdev = std::sqrt(v_n);
            }
        }
    }
    spdlog::debug("Mean {}, stdev {}", x_n, stdev);
    for (int i = signal_end; (fwd ? i > 0 : i < read->raw_data.size(0)); (fwd ? i-- : i++)) {
        float x = signal[i];
        if (std::abs(x - x_n) > 1 * stdev) {
            spdlog::debug("Reached end at {}", i);
            break;
        }
        signal_start = i;
    }
    if (!fwd) {
        std::swap(signal_start, signal_end);
    }
    return {signal_start, signal_end};
}

int estimate_samples_per_base(const std::vector<uint64_t>& seq_to_sig_map,
                              const std::string& seq,
                              bool fwd,
                              int signal_start,
                              int signal_end) {
    int c = 0;
    int j = 0;
    const char s = (fwd ? 'A' : 'T');
    int s_i = -1, e_i = -1;
    for (int i = 0; i < seq.length(); i++) {
        char cur_char = seq[i];
        int nt_signal_start = seq_to_sig_map[i];
        int nt_signal_end = seq_to_sig_map[i + 1];
        if (fwd && nt_signal_end >= signal_start) {
            continue;
        }
        if (!fwd && nt_signal_start < signal_end) {
            continue;
        }
        if (i < 10) {
            continue;
        }
        if (s_i < 0)
            s_i = i;
        e_i = i;
        c += seq_to_sig_map[i + 1] - seq_to_sig_map[i];
        j++;
    }
    float num_samples_per_base = static_cast<float>(c) / j;
    spdlog::debug("Using {} samples to estimate samples/base in range {} {}", j, s_i, e_i);
    return static_cast<int>(num_samples_per_base);
}

int estimate_samples_per_base(const dorado::ReadPtr& read) {
    float num_samples_per_base = static_cast<float>(read->raw_data.size(0)) / read->seq.length();
    return static_cast<int>(num_samples_per_base);
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

            // Walk through signal
            const c10::Half* signal = static_cast<c10::Half*>(read->raw_data.data_ptr());

            auto num_samples_per_base = estimate_samples_per_base(read);
            //auto num_samples_per_base = estimate_samples_per_base(seq_to_sig_map, read->seq, fwd, signal_start, signal_end);

            auto [signal_start, signal_end] = determine_signal_bounds4(
                    signal_anchor, signal, seq_to_sig_map, fwd, read, num_samples_per_base);

            int num_bases = static_cast<int>((signal_end - signal_start) / num_samples_per_base);
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
