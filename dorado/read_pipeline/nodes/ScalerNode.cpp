#include "read_pipeline/nodes/ScalerNode.h"

#include "config/BasecallModelConfig.h"
#include "demux/adapter_info.h"
#include "models/kits.h"
#include "torch_utils/tensor_utils.h"
#include "torch_utils/trim.h"
#include "utils/context_container.h"
#include "utils/log_utils.h"

#include <ATen/Functions.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorOperators.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <unordered_map>
#include <utility>

static constexpr float EPS = 1e-9f;

using Slice = at::indexing::Slice;

namespace {

using namespace dorado::config;

std::pair<float, float> med_mad(const at::Tensor& x) {
    // See https://en.wikipedia.org/wiki/Median_absolute_deviation
    //  (specifically the "Relation to standard deviation" section)
    constexpr float factor = 1.4826f;
    //Calculate signal median and median absolute deviation
    auto med = x.median();
    auto mad = at::median(at::abs(x - med)) * factor + EPS;
    return {med.item<float>(), mad.item<float>()};
}

std::pair<float, float> normalisation(const QuantileScalingParams& params, const at::Tensor& x) {
    // Calculate shift and scale factors for normalisation.
    auto quantiles =
            dorado::utils::quantile_counting(x, at::tensor({params.quantile_a, params.quantile_b}));
    float q_a = quantiles[0].item<float>();
    float q_b = quantiles[1].item<float>();
    float shift = std::max(10.0f, params.shift_multiplier * (q_a + q_b));
    float scale = std::max(1.0f, params.scale_multiplier * (q_b - q_a));
    return {shift, scale};
}

using SampleType = dorado::models::SampleType;

// This function returns the approximate position where the DNA adapter
// in a dRNA read ends. The adapter location is determined by looking
// at the median signal value over a sliding window on the raw signal.
int determine_rna_adapter_pos(const dorado::SimplexRead& read) {
    assert(read.read_common.raw_data.dtype() == at::kShort);

    const int kWindowSize = 250;
    const int kStride = 50;
    const int16_t kMedianDiff = 125;
    const int16_t kMedianDiffForDiffOnlyCheck = 150;

    const int16_t kMinMedianForRNASignal = 700;

    int signal_len = static_cast<int>(read.read_common.get_raw_data_samples());
    const int16_t* signal = static_cast<int16_t*>(read.read_common.raw_data.data_ptr());

    // Check the median value change over 5 windows.
    std::array<int16_t, 5> medians = {0, 0, 0, 0, 0};
    std::array<int32_t, 5> window_pos = {0, 0, 0, 0, 0};
    int median_pos = 0;
    int break_point = 0;
    const int signal_start = 1000;
    const int signal_end = 3 * signal_len / 4;
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
        // The range of raw signal values is within the range of [-500, 3000] (TODO: they're
        // likely are non-negative but need to confirm that). So the median values lie
        // in the same range, and any difference between the median values
        // will not result in an overflow with the int16_t data type.
        int16_t min_median = *minmax.first;
        int16_t max_median = *minmax.second;
        auto min_pos = std::distance(medians.begin(), minmax.first);
        auto max_pos = std::distance(medians.begin(), minmax.second);

        dorado::utils::trace_log("window {}-{} min {} max {} diff {}", i, i + kWindowSize,
                                 min_median, max_median, (max_median - min_median));

        if ((median_pos >= static_cast<int>(medians.size()) &&
             window_pos[max_pos] > window_pos[min_pos]) &&
            (((max_median > kMinMedianForRNASignal) && (max_median - min_median > kMedianDiff)) ||
             (max_median - min_median > kMedianDiffForDiffOnlyCheck))) {
            break_point = i;
            break;
        }
        ++median_pos;
    }

    return break_point;
}

constexpr float pore_level_kit14_prom = 199.21f;
constexpr float pore_level_kit14_min = 197.61f;
constexpr float pore_level_rna004_prom = 194.97f;
constexpr float pore_level_rna004_min = 195.50f;
constexpr float pore_level_flongle = 200.0f;

using FC = dorado::models::Flowcell;
std::unordered_map<FC, float> flowcell_expected_pore_levels{
        {FC::FLO_FLG114, pore_level_flongle},      {FC::FLO_FLG114HD, pore_level_flongle},
        {FC::FLO_MIN004RA, pore_level_rna004_min}, {FC::FLO_PRO004RA, pore_level_rna004_prom},
        {FC::FLO_MIN114, pore_level_kit14_min},    {FC::FLO_MIN114HD, pore_level_kit14_min},
        {FC::FLO_PRO114, pore_level_kit14_prom},   {FC::FLO_PRO114HD, pore_level_kit14_prom},
        {FC::FLO_PRO114M, pore_level_kit14_prom},
};

std::optional<float> get_expected_open_pore_level(const std::string& flow_cell_code) {
    auto pore_level =
            flowcell_expected_pore_levels.find(dorado::models::flowcell_code(flow_cell_code));
    if (pore_level == flowcell_expected_pore_levels.end()) {
        return std::nullopt;
    }
    return pore_level->second;
}

}  // anonymous namespace

namespace dorado {

void ScalerNode::input_thread_fn() {
    at::InferenceMode inference_mode_guard;

    Message message;
    while (get_input_message(message)) {
        // If this message isn't a Simplex read, just forward it to the sink.
        if (!std::holds_alternative<SimplexReadPtr>(message)) {
            send_message_to_sink(std::move(message));
            continue;
        }

        auto read = std::get<SimplexReadPtr>(std::move(message));

        bool is_rna_model = m_model_type == SampleType::RNA004;

        // Trim adapter for RNA first before scaling.
        int trim_start = 0;
        if (is_rna_model) {
            std::shared_ptr<const demux::AdapterInfo> adapter_info =
                    read->read_common.client_info ? read->read_common.client_info->contexts()
                                                            .get_ptr<const demux::AdapterInfo>()
                                                  : nullptr;

            const bool has_rna_based_adapters = adapter_info && adapter_info->rna_adapters;
            if (!has_rna_based_adapters) {
                trim_start = determine_rna_adapter_pos(*read);
                if (size_t(trim_start) < read->read_common.get_raw_data_samples()) {
                    read->read_common.raw_data = read->read_common.raw_data.index(
                            {Slice(trim_start, at::indexing::None)});
                    read->read_common.rna_adapter_end_signal_pos = 0;
                } else {
                    // If RNA adapter isn't trimmed, track where the adapter signal is ending
                    // so it can be used during polyA estimation.
                    read->read_common.rna_adapter_end_signal_pos = trim_start;
                    // Since we're not actualy trimming the signal, reset the trim value to 0.
                    trim_start = 0;
                }
            }
        }

        assert(read->read_common.raw_data.dtype() == at::kShort);

        float scale = 1.0f;
        float shift = 0.0f;
        float open_pore_adjustment = 0.0f;

        read->read_common.scaling_method = to_string(m_scaling_params.strategy);
        if (m_scaling_params.strategy == ScalingStrategy::PA) {
            // We want to keep the scaling formula `(x - shift) / scale` consistent between
            // quantile and pA methods as this affects downstream tools.
            const auto& stdn = m_scaling_params.standardisation;
            if (stdn.standardise) {
                // Standardise from scaled pa
                // 1. x_pa  = (Scale)*(x + Offset)
                // 2. x_std = (1 / Stdev)*(x_pa - Mean)
                // => x_std = (Scale / Stdev)*(x + (Offset - (Mean / Scale)))
                // => x_std = (x - ((Mean / Scale) - Offset)) / (Stdev / Scale)
                scale = stdn.stdev / read->scaling;
                shift = (stdn.mean / read->scaling) - read->offset;
            } else {
                scale = 1.f / read->scaling;
                shift = -1.f * read->offset;
            }

            if (!std::isnan(read->open_pore_level)) {
                std::optional<float> expected_open_pore_level =
                        get_expected_open_pore_level(read->read_common.flow_cell_product_code);
                if (expected_open_pore_level.has_value() && *expected_open_pore_level != 0) {
                    open_pore_adjustment =
                            (read->open_pore_level - *expected_open_pore_level) / read->scaling;
                }
            }
        } else {
            // Ignore the RNA adapter. If this is DNA or we've already trimmed the adapter, this will be zero
            auto scaling_data = read->read_common.raw_data.index(
                    {Slice(read->read_common.rna_adapter_end_signal_pos, at::indexing::None)});
            std::tie(shift, scale) =
                    m_scaling_params.strategy == ScalingStrategy::QUANTILE
                            ? normalisation(m_scaling_params.quantile, scaling_data)
                            : med_mad(scaling_data);
        }

        // raw_data comes from DataLoader with dtype int16.  We send it on as float16 after
        // shifting/scaling in float32 form.
        utils::shift_scale_tensor_i16_to_f16_inplace(read->read_common.raw_data,
                                                     shift + open_pore_adjustment, scale);

        // move the shift and scale into pA.
        read->read_common.scale = read->scaling * scale;
        read->read_common.shift = read->scaling * (shift + read->offset);

        // Don't perform DNA trimming on RNA since it looks too different and we lose useful signal.
        if (!is_rna_model) {
            if (trim_start == 0 && m_scaling_params.standardisation.standardise) {
                // Constant trimming level for standardised scaling
                // In most cases kit14 trim algorithm returns 10, so bypassing the heuristic
                // and applying 10 for pA scaled data.
                // TODO: may need refinement in the future
                trim_start = 10;
            } else if (trim_start == 0) {
                // 8000 value may be changed in future. Currently this is found to work well.
                int max_samples = std::min(
                        8000, static_cast<int>(read->read_common.get_raw_data_samples() / 2));
                trim_start = utils::trim(
                        read->read_common.raw_data.index({Slice(at::indexing::None, max_samples)}),
                        utils::DEFAULT_TRIM_THRESHOLD, utils::DEFAULT_TRIM_WINDOW_SIZE,
                        utils::DEFAULT_TRIM_MIN_ELEMENTS);
            }

            if (size_t(trim_start) < read->read_common.get_raw_data_samples()) {
                read->read_common.raw_data =
                        read->read_common.raw_data.index({Slice(trim_start, at::indexing::None)});
            } else {
                trim_start = 0;
            }
        }

        read->read_common.num_trimmed_samples = trim_start;

        dorado::utils::trace_log("ScalerNode: {} shift: {} scale: {} trim: {}",
                                 read->read_common.read_id, shift, scale, trim_start);

        // Pass the read to the next node
        send_message_to_sink(std::move(read));
    }
}

ScalerNode::ScalerNode(const SignalNormalisationParams& config,
                       SampleType model_type,
                       int num_worker_threads,
                       size_t max_reads)
        : MessageSink(max_reads, num_worker_threads),
          m_scaling_params(config),
          m_model_type(model_type) {}

ScalerNode::~ScalerNode() { stop_input_processing(utils::AsyncQueueTerminateFast::Yes); }

std::string ScalerNode::get_name() const { return "ScalerNode"; }

void ScalerNode::terminate(const TerminateOptions& terminate_options) {
    stop_input_processing(terminate_options.fast);
}

void ScalerNode::restart() {
    start_input_processing([this] { input_thread_fn(); }, "scaler_node");
}

}  // namespace dorado
