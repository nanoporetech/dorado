#include "ScalerNode.h"

#include "basecall/CRFModelConfig.h"
#include "utils/tensor_utils.h"
#include "utils/trim.h"

#include <ATen/ATen.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <utility>

static constexpr float EPS = 1e-9f;

using namespace std::chrono_literals;
using Slice = at::indexing::Slice;

namespace dorado {
using SampleType = basecall::SampleType;
using ScalingStrategy = basecall::ScalingStrategy;
using SignalNormalisationParams = basecall::SignalNormalisationParams;

std::pair<float, float> ScalerNode::normalisation(const at::Tensor& x) {
    // Calculate shift and scale factors for normalisation.
    const auto& params = m_scaling_params.quantile;
    auto quantiles =
            dorado::utils::quantile_counting(x, at::tensor({params.quantile_a, params.quantile_b}));
    float q_a = quantiles[0].item<float>();
    float q_b = quantiles[1].item<float>();
    float shift = std::max(10.0f, params.shift_multiplier * (q_a + q_b));
    float scale = std::max(1.0f, params.scale_multiplier * (q_b - q_a));
    return {shift, scale};
}

std::pair<float, float> ScalerNode::med_mad(const at::Tensor& x) {
    // See https://en.wikipedia.org/wiki/Median_absolute_deviation
    //  (specifically the "Relation to standard deviation" section)
    constexpr float factor = 1.4826f;
    //Calculate signal median and median absolute deviation
    auto med = x.median();
    auto mad = at::median(at::abs(x - med)) * factor + EPS;
    return {med.item<float>(), mad.item<float>()};
}

// This function returns the approximate position where the DNA adapter
// in a dRNA read ends. The adapter location is determined by looking
// at the median signal value over a sliding window on the raw signal.
// RNA002 and RNA004 have different offsets and thresholds for the
// sliding window heuristic.
int determine_rna_adapter_pos(const dorado::SimplexRead& read, dorado::SampleType model_type) {
    assert(read.read_common.raw_data.dtype() == at::kShort);
    static const std::unordered_map<dorado::SampleType, int> kOffsetMap = {
            {dorado::SampleType::RNA002, 3500}, {dorado::SampleType::RNA004, 1000}};
    static const std::unordered_map<dorado::SampleType, int16_t> kAdapterCutoff = {
            {dorado::SampleType::RNA002, 550}, {dorado::SampleType::RNA004, 700}};

    const int kWindowSize = 250;
    const int kStride = 50;
    const int16_t kMedianDiff = 125;

    const int16_t kMinMedianForRNASignal = kAdapterCutoff.at(model_type);

    int signal_len = int(read.read_common.get_raw_data_samples());
    const int16_t* signal = static_cast<int16_t*>(read.read_common.raw_data.data_ptr());

    // Check the median value change over 5 windows.
    std::array<int16_t, 5> medians = {0, 0, 0, 0, 0};
    std::array<int32_t, 5> window_pos = {0, 0, 0, 0, 0};
    int median_pos = 0;
    int break_point = 0;
    const int signal_start = kOffsetMap.at(model_type);
    const int signal_end = static_cast<int>(3 * signal_len / 4);
    for (int i = signal_start; i < signal_end; i += kStride) {
        auto slice = at::from_blob(const_cast<int16_t*>(&signal[i]),
                                   {static_cast<int>(std::min(kWindowSize, signal_len - i))},
                                   at::TensorOptions().dtype(at::kShort));
        int16_t median = slice.median().item<int16_t>();
        medians[median_pos % medians.size()] = median;
        // Since the medians are stored in a circular buffer, we need
        // to store the actual window positions for the median values
        // as well to check that maximum median value came from a window
        // after that of the minimum median value.
        window_pos[median_pos % window_pos.size()] = median_pos;
        auto minmax = std::minmax_element(medians.begin(), medians.end());
        int16_t min_median = *minmax.first;
        int16_t max_median = *minmax.second;
        auto min_pos = std::distance(medians.begin(), minmax.first);
        auto max_pos = std::distance(medians.begin(), minmax.second);
        spdlog::trace("window {}-{} min {} max {} diff {}", i, i + kWindowSize, min_median,
                      max_median, (max_median - min_median));
        if ((median_pos >= int(medians.size())) && (max_median > kMinMedianForRNASignal) &&
            (max_median - min_median > kMedianDiff) &&
            (window_pos[max_pos] > window_pos[min_pos])) {
            break_point = i;
            break;
        }
        ++median_pos;
    }

    return break_point;
}

void ScalerNode::worker_thread() {
    at::InferenceMode inference_mode_guard;

    Message message;
    while (get_input_message(message)) {
        // If this message isn't a Simplex read, just forward it to the sink.
        if (!std::holds_alternative<SimplexReadPtr>(message)) {
            send_message_to_sink(std::move(message));
            continue;
        }

        auto read = std::get<SimplexReadPtr>(std::move(message));

        bool is_rna = (m_model_type == SampleType::RNA002 || m_model_type == SampleType::RNA004);
        // Trim adapter for RNA first before scaling.
        int trim_start = 0;
        if (is_rna) {
            trim_start = determine_rna_adapter_pos(*read, m_model_type);
            if (m_trim_adapter) {
                read->read_common.raw_data =
                        read->read_common.raw_data.index({Slice(trim_start, at::indexing::None)});
                read->read_common.rna_adapter_end_signal_pos = 0;
            } else {
                // If RNA adapter isn't trimmed, track where the adapter signal is ending
                // so it can be used during polyA estimation.
                read->read_common.rna_adapter_end_signal_pos = trim_start;
            }
        }

        assert(read->read_common.raw_data.dtype() == at::kShort);

        float scale = 1.0f;
        float shift = 0.0f;

        read->read_common.scaling_method = to_string(m_scaling_params.strategy);
        if (m_scaling_params.strategy == ScalingStrategy::PA) {
            const auto& stdn = m_scaling_params.standarisation;
            if (stdn.standardise) {
                // Standardise from scaled pa
                // 1. x_pa  = (Scale)*(x + Offset)
                // 2. x_std = (1 / Stdev)*(x_pa - Mean)
                // => x_std = (Scale / Stdev)*(x + (Offset - (Mean / Scale)))
                //            ---- scale ---        ------- shift --------
                scale = read->scaling / stdn.stdev;
                shift = read->offset - (stdn.mean / read->scaling);
            } else {
                scale = read->scaling;
                shift = read->offset;
            }

            read->read_common.raw_data =
                    ((read->read_common.raw_data.to(at::kFloat) + shift) * scale)
                            .to(at::ScalarType::Half);

            read->read_common.scale = scale;
            read->read_common.shift = shift;
        } else {
            std::tie(shift, scale) = m_scaling_params.strategy == ScalingStrategy::QUANTILE
                                             ? normalisation(read->read_common.raw_data)
                                             : med_mad(read->read_common.raw_data);

            // raw_data comes from DataLoader with dtype int16.  We send it on as float16 after
            // shifting/scaling in float32 form.
            read->read_common.raw_data =
                    ((read->read_common.raw_data.to(at::kFloat) - shift) / scale)
                            .to(at::ScalarType::Half);
            // move the shift and scale into pA.
            read->read_common.scale = read->scaling * scale;
            read->read_common.shift = read->scaling * (shift + read->offset);
        }

        // Don't perform DNA trimming on RNA since it looks too different and we lose useful signal.
        if (!is_rna) {
            if (m_scaling_params.standarisation.standardise) {
                // Constant trimming level for standardised scaling
                // In most cases kit14 trim algorithm returns 10, so bypassing the heuristic
                // and applying 10 for pA scaled data.
                // TODO: may need refinement in the future
                trim_start = 10;
            } else {
                // 8000 value may be changed in future. Currently this is found to work well.
                int max_samples = std::min(
                        8000, static_cast<int>(read->read_common.get_raw_data_samples() / 2));
                trim_start = utils::trim(
                        read->read_common.raw_data.index({Slice(at::indexing::None, max_samples)}),
                        utils::DEFAULT_TRIM_THRESHOLD, utils::DEFAULT_TRIM_WINDOW_SIZE,
                        utils::DEFAULT_TRIM_MIN_ELEMENTS);
            }

            read->read_common.raw_data =
                    read->read_common.raw_data.index({Slice(trim_start, at::indexing::None)});
        }

        read->read_common.num_trimmed_samples = trim_start;

        spdlog::trace("ScalerNode: {} shift: {} scale: {} trim: {}", read->read_common.read_id,
                      shift, scale, trim_start);

        // Pass the read to the next node
        send_message_to_sink(std::move(read));
    }
}

ScalerNode::ScalerNode(const SignalNormalisationParams& config,
                       SampleType model_type,
                       bool trim_adapter,
                       int num_worker_threads,
                       size_t max_reads)
        : MessageSink(max_reads),
          m_num_worker_threads(num_worker_threads),
          m_scaling_params(config),
          m_model_type(model_type),
          m_trim_adapter(trim_adapter) {
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
