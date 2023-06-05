#include "ScalerNode.h"

#include "utils/tensor_utils.h"
#include "utils/trim.h"

#include <algorithm>
#include <chrono>
#include <utility>

using namespace std::chrono_literals;
using Slice = torch::indexing::Slice;

namespace {

std::pair<float, float> normalisation(torch::Tensor& x) {
    // Calculate shift and scale factors for normalisation.
    auto quantiles = dorado::utils::quantile_counting(x, torch::tensor({0.2, 0.9}));
    float q20 = quantiles[0].item<float>();
    float q90 = quantiles[1].item<float>();
    float shift = std::max(10.0f, 0.51f * (q20 + q90));
    float scale = std::max(1.0f, 0.53f * (q90 - q20));
    return {shift, scale};
}

}  // namespace

namespace dorado {

void ScalerNode::worker_thread() {
    Message message;
    while (m_work_queue.try_pop(message)) {
        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto read = std::get<std::shared_ptr<Read>>(message);

        const auto [shift, scale] = normalisation(read->raw_data);
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
        m_sink.push_message(read);
    }

    int num_worker_threads = --m_num_worker_threads;
    if (num_worker_threads == 0) {
        m_sink.terminate();
    }
}

ScalerNode::ScalerNode(MessageSink& sink, int num_worker_threads, size_t max_reads)
        : MessageSink(max_reads), m_sink(sink), m_num_worker_threads(num_worker_threads) {
    for (int i = 0; i < m_num_worker_threads; i++) {
        std::unique_ptr<std::thread> scaler_worker_thread =
                std::make_unique<std::thread>(&ScalerNode::worker_thread, this);
        worker_threads.push_back(std::move(scaler_worker_thread));
    }
}

ScalerNode::~ScalerNode() {
    terminate();

    // Wait for all the Scaler Node's worker threads to terminate
    for (auto& t : worker_threads) {
        t->join();
    }

    // Notify the sink that the Scaler Node has terminated
    m_sink.terminate();
}

stats::NamedStats ScalerNode::sample_stats() const { return stats::from_obj(m_work_queue); }

}  // namespace dorado
