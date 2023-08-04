#include "PolyACalculator.h"

#include "3rdparty/edlib/edlib/include/edlib.h"
#include "utils/sequence_utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>

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

        int window_size = 150;
        std::string read_top = read->seq.substr(0, window_size);
        auto bottom_start = std::max(0, (int)read->seq.length() - window_size);
        std::string read_bottom = read->seq.substr(bottom_start, window_size);

        const std::string SSP = "TTTCTGTTGGTGCTGATATTGCTTT";
        const std::string SSP_rc = utils::reverse_complement(SSP);
        const std::string VNP = "ACTTGCCTGTCGCTCTATCTTCAGAGGAGAGTCCGCCGCCCGCAAGTTTT";
        const std::string VNP_rc = utils::reverse_complement(VNP);

        const auto stride = read->model_stride;
        const auto seq_to_sig_map = utils::moves_to_map(read->moves, stride, read->raw_data.size(0),
                                                        read->seq.size() + 1);

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

        bool fwd = true;
        int start = 0, end = 0;
        if (dist_v2 < dist_v1) {
            fwd = false;
            start = top_v2.endLocations[0];
        } else {
            end = bottom_start + bottom_v1.startLocations[0];
        }

        int signal_end = fwd ? seq_to_sig_map[end + 1] : seq_to_sig_map[start];
        spdlog::debug("Strand {}; poly A/T signal begin {}", fwd ? '+' : '-', signal_end);

        // Walk through signal

        float x_n = 0, x_n_1 = 0;
        float v_n = 1, v_n_1 = 0;
        int n = 0;
        int signal_start = 0;
        const c10::Half* signal = static_cast<c10::Half*>(read->raw_data.data_ptr());
        for (int i = signal_end; (fwd ? i > 0 : i < read->raw_data.size(0)); (fwd ? i-- : i++)) {
            x_n_1 = x_n;
            v_n_1 = v_n;
            float x = signal[i];
            auto stdev = std::sqrt(v_n_1);
            spdlog::debug("input {}, Mean {} stddev {}", x, x_n, stdev);
            if (n > 10 and x > x_n_1 + 4 * stdev) {
                spdlog::debug("Reached end at {}", i);
                break;
            }
            x_n = x_n_1 + float(x - x_n_1) / (n + 1);
            v_n = v_n_1 + float((x - x_n_1) * (x - x_n) - v_n_1) / (n + 1);
            n++;
            signal_start = i;
        }
        if (!fwd) {
            std::swap(signal_start, signal_end);
        }
        float num_samples_per_base =
                static_cast<float>(read->raw_data.size(0)) / read->seq.length();
        const int kNumBases = 200;
        //int c = 0;
        //int j = 0;
        //for(int i = 0; i < read->seq.length() && j < kNumBases; i++) {
        //    if (read->seq[i] == (fwd ? 'A' : 'T')) {
        //        c += seq_to_sig_map[i + 1] - seq_to_sig_map[i];
        //        j++;
        //    }
        //}
        //float num_samples_per_base = static_cast<float>(c) / j;
        //int start_base = (read->seq.length() >> 1) - (kNumBases >> 1);
        //float num_samples_per_base = static_cast<float>(seq_to_sig_map[start_base + kNumBases + 1] - seq_to_sig_map[start_base]) / kNumBases;
        int num_bases = static_cast<int>((signal_end - signal_start) / num_samples_per_base);
        if (num_bases < 50) {
            spdlog::debug("{} PolyA bases {}, Signal range is {} {}, primer {}", read->read_id,
                          num_bases, signal_start, signal_end, end);
            spdlog::debug("Region is {}", read->seq.substr(end - num_bases, num_bases));
        } else {
            spdlog::warn("{} PolyA bases {}, Signal range is {} {} primer {}", read->read_id,
                         num_bases, signal_start, signal_end, end);
        }

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
}

void PolyACalculator::restart() {
    restart_input_queue();
    start_threads();
}

}  // namespace dorado
