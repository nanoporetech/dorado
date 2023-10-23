#include "ScalerNode.h"

#include "utils/tensor_utils.h"
#include "utils/trim.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <utility>

#define EPS 1e-9f

using namespace std::chrono_literals;
using Slice = torch::indexing::Slice;

namespace dorado {

std::pair<float, float> ScalerNode::normalisation(const torch::Tensor& x) {
    // Calculate shift and scale factors for normalisation.
    auto quantiles = dorado::utils::quantile_counting(
            x, torch::tensor({m_scaling_params.quantile_a, m_scaling_params.quantile_b}));
    float q_a = quantiles[0].item<float>();
    float q_b = quantiles[1].item<float>();
    float shift = std::max(10.0f, m_scaling_params.shift_multiplier * (q_a + q_b));
    float scale = std::max(1.0f, m_scaling_params.scale_multiplier * (q_b - q_a));
    return {shift, scale};
}

std::pair<float, float> ScalerNode::med_mad(const torch::Tensor& x) {
    // See https://en.wikipedia.org/wiki/Median_absolute_deviation
    //  (specifically the "Relation to standard deviation" section)
    constexpr float factor = 1.4826;
    //Calculate signal median and median absolute deviation
    auto med = x.median();
    auto mad = torch::median(torch::abs(x - med)) * factor + EPS;
    return {med.item<float>(), mad.item<float>()};
}

// This function returns the approximate position where the DNA adapter
// in a dRNA read ends. The adapter location is determined by looking
// at the median signal value of a sliding window over the raw signal.
// RNA002 and RNA004 have different offsets and thresholds for the
// sliding window heuristic.
int determine_rna_adapter_pos(const dorado::SimplexRead& read, dorado::SampleType model_type) {
    static const std::unordered_map<dorado::SampleType, int> kOffsetMap = {
            {dorado::SampleType::RNA002, 5000}, {dorado::SampleType::RNA004, 1000}};
    static const std::unordered_map<dorado::SampleType, int> kMaxSignalPosMap = {
            {dorado::SampleType::RNA002, 15000}, {dorado::SampleType::RNA004, 5000}};
    static const std::unordered_map<dorado::SampleType, int16_t> kAdapterCutoff = {
            {dorado::SampleType::RNA002, 550}, {dorado::SampleType::RNA004, 825}};

    const int kWindowSize = 250;
    const int kStride = 50;
    const int kOffset = kOffsetMap.at(model_type);
    const int kMaxSignalPos = kMaxSignalPosMap.at(model_type);

    const int16_t kMinMedianForRNASignal = kAdapterCutoff.at(model_type);

    int bp = 0;
    int signal_len = read.read_common.get_raw_data_samples();
    auto sig_fp32 = read.read_common.raw_data.to(torch::kInt16);
    int16_t last_median = 0;
    for (int i = kOffset; i < std::min(signal_len / 2, kMaxSignalPos); i += kStride) {
        auto slice = sig_fp32.slice(0, i, std::min(signal_len, i + kWindowSize));
        int16_t median = slice.median().item<int16_t>();
        if (i > kOffset && median > kMinMedianForRNASignal && (median - last_median > 75)) {
            bp = i;
            break;
        }
        last_median = median;
    }

    spdlog::debug("Approx break point {}", bp);

    return bp;
}

void ScalerNode::worker_thread() {
    torch::InferenceMode inference_mode_guard;

    Message message;
    while (get_input_message(message)) {
        // If this message isn't a Simplex read, just forward it to the sink.
        if (!std::holds_alternative<SimplexReadPtr>(message)) {
            send_message_to_sink(std::move(message));
            continue;
        }

        auto read = std::get<SimplexReadPtr>(std::move(message));

        // Trim adapter for RNA first before scaling.
        int trim_start = 0;
        if (m_is_rna) {
            trim_start = determine_rna_adapter_pos(*read, m_model_type);
            read->read_common.raw_data =
                    read->read_common.raw_data.index({Slice(trim_start, torch::indexing::None)});
        }

        assert(read->read_common.raw_data.dtype() == torch::kInt16);
        auto [shift, scale] = m_scaling_params.quantile_scaling
                                      ? normalisation(read->read_common.raw_data)
                                      : med_mad(read->read_common.raw_data);
        read->read_common.scaling_method =
                m_scaling_params.quantile_scaling ? "quantile" : "med_mad";

        // raw_data comes from DataLoader with dtype int16.  We send it on as float16 after
        // shifting/scaling in float32 form.
        read->read_common.raw_data =
                ((read->read_common.raw_data.to(torch::kFloat) - shift) / scale)
                        .to(torch::kFloat16);

        // move the shift and scale into pA.
        read->read_common.scale = read->scaling * scale;
        read->read_common.shift = read->scaling * (shift + read->offset);

        // Don't perform DNA trimming on RNA since it looks too different and we lose useful signal.
        if (!m_is_rna) {
            // 8000 value may be changed in future. Currently this is found to work well.
            int max_samples =
                    std::min(8000, static_cast<int>(read->read_common.get_raw_data_samples() / 2));
            trim_start = utils::trim(
                    read->read_common.raw_data.index({Slice(torch::indexing::None, max_samples)}));
            read->read_common.raw_data =
                    read->read_common.raw_data.index({Slice(trim_start, torch::indexing::None)});
        }

        read->read_common.num_trimmed_samples = trim_start;

        spdlog::debug("{} {} {} {}", read->read_common.read_id, shift, scale, trim_start);

        // Pass the read to the next node
        send_message_to_sink(std::move(read));
    }
}

ScalerNode::ScalerNode(const SignalNormalisationParams& config,
                       SampleType model_type,
                       int num_worker_threads,
                       size_t max_reads)
        : MessageSink(max_reads),
          m_scaling_params(config),
          m_num_worker_threads(num_worker_threads),
          m_is_rna(model_type == SampleType::RNA002 || model_type == SampleType::RNA004),
          m_model_type(model_type) {
    start_threads();
}

void ScalerNode::start_threads() {
    for (int i = 0; i < m_num_worker_threads; i++) {
        std::unique_ptr<std::thread> scaler_worker_thread =
                std::make_unique<std::thread>(&ScalerNode::worker_thread, this);
        m_worker_threads.push_back(std::move(scaler_worker_thread));
    }
}

void ScalerNode::terminate_impl() {
    terminate_input_queue();

    // Wait for all the Scaler Node's worker threads to terminate
    for (auto& t : m_worker_threads) {
        if (t->joinable()) {
            t->join();
        }
    }
    m_worker_threads.clear();
}

void ScalerNode::restart() {
    restart_input_queue();
    start_threads();
}

stats::NamedStats ScalerNode::sample_stats() const { return stats::from_obj(m_work_queue); }

}  // namespace dorado
