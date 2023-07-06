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

void ScalerNode::worker_thread() {
    Message message;
    while (m_work_queue.try_pop(message)) {
        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto read = std::get<std::shared_ptr<Read>>(message);
        assert(read->raw_data.dtype() == torch::kInt16);
        const auto [shift, scale] = m_scaling_params.quantile_scaling
                                            ? normalisation(read->raw_data)
                                            : med_mad(read->raw_data);
        read->scaling_method = m_scaling_params.quantile_scaling ? "quantile" : "med_mad";

        // raw_data comes from DataLoader with dtype int16.  We send it on as float16 after
        // shifting/scaling in float32 form.
        read->raw_data = ((read->raw_data.to(torch::kFloat) - shift) / scale).to(torch::kFloat16);

        // move the shift and scale into pA.
        read->scale = read->scaling * scale;
        read->shift = read->scaling * (shift + read->offset);

        // 8000 value may be changed in future. Currently this is found to work well.
        int max_samples = std::min(8000, static_cast<int>(read->raw_data.size(0) / 2));
        int trim_start =
                utils::trim(read->raw_data.index({Slice(torch::indexing::None, max_samples)}));

        read->raw_data = read->raw_data.index({Slice(trim_start, torch::indexing::None)});
        read->num_trimmed_samples = trim_start;

        // Pass the read to the next node
        send_message_to_sink(read);
    }

    int num_worker_threads = --m_num_worker_threads;
}

ScalerNode::ScalerNode(const SignalNormalisationParams& config,
                       int num_worker_threads,
                       size_t max_reads)
        : MessageSink(max_reads),
          m_scaling_params(config),
          m_num_worker_threads(num_worker_threads) {
    for (int i = 0; i < m_num_worker_threads; i++) {
        std::unique_ptr<std::thread> scaler_worker_thread =
                std::make_unique<std::thread>(&ScalerNode::worker_thread, this);
        worker_threads.push_back(std::move(scaler_worker_thread));
    }
}

void ScalerNode::terminate_impl() {
    terminate_input_queue();

    // Wait for all the Scaler Node's worker threads to terminate
    for (auto& t : worker_threads) {
        if (t->joinable()) {
            t->join();
        }
    }
}

stats::NamedStats ScalerNode::sample_stats() const { return stats::from_obj(m_work_queue); }

}  // namespace dorado
