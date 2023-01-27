#include "ScalerNode.h"

#include "utils/tensor_utils.h"

#include <algorithm>
#include <chrono>

using namespace std::chrono_literals;

namespace {

std::pair<float, float> normalisation(torch::Tensor& x) {
    //Calculate shift and scale factors for normalisation.
    auto quantiles = dorado::utils::quantile_counting(x, torch::tensor({0.2, 0.9}));
    float q20 = quantiles[0].item<float>();
    float q90 = quantiles[1].item<float>();
    float shift = std::max(10.0f, 0.51f * (q20 + q90));
    float scale = std::max(1.0f, 0.53f * (q90 - q20));
    return std::make_pair(shift, scale);
}

}  // namespace

namespace dorado {

void ScalerNode::worker_thread() {
    while (true) {
        // Wait until we are provided with a read
        std::unique_lock<std::mutex> lock(m_cv_mutex);
        m_cv.wait_for(lock, 100ms, [this] { return !m_reads.empty(); });
        if (m_reads.empty()) {
            if (m_terminate) {
                // Termination flag is set and read input queue is empty, so terminate the worker
                return;
            } else {
                continue;
            }
        }

        std::shared_ptr<Read> read = m_reads.front();
        m_reads.pop_front();
        lock.unlock();

        auto [shift, scale] = normalisation(read->raw_data);
        read->raw_data = (read->raw_data - shift) / scale;

        // move the shift and scale into pA.
        read->scale = read->scaling * scale;
        read->shift = read->scaling * (shift + read->offset);

        float threshold = read->shift + read->scale * 2.4;

        // 8000 value may be changed in future. Currently this is found to work well.
        int trim_start =
                trim(read->raw_data.index({torch::indexing::Slice(torch::indexing::None, 8000)}),
                     threshold);

        read->raw_data =
                read->raw_data.index({torch::indexing::Slice(trim_start, torch::indexing::None)});
        read->num_trimmed_samples = trim_start;

        // Pass the read to the next node
        m_sink.push_read(read);
    }
}

ScalerNode::ScalerNode(ReadSink& sink, int num_worker_threads, size_t max_reads)
        : ReadSink(max_reads), m_sink(sink), m_num_worker_threads(num_worker_threads) {
    for (int i = 0; i < m_num_worker_threads; i++) {
        std::unique_ptr<std::thread> scaler_worker_thread =
                std::make_unique<std::thread>(&ScalerNode::worker_thread, this);
        worker_threads.push_back(std::move(scaler_worker_thread));
    }
}

ScalerNode::~ScalerNode() {
    terminate();
    m_cv.notify_one();

    // Wait for all the Scaler Node's worker threads to terminate
    for (auto& t : worker_threads) {
        t->join();
    }

    //Notify the sink that the Scaler Node has terminated
    m_sink.terminate();
}

int ScalerNode::trim(torch::Tensor signal,
                     int window_size,
                     float threshold,
                     int min_elements,
                     int max_samples,
                     float max_trim) {
    int min_trim = 10;
    bool seen_peak = false;
    int num_samples = std::min(max_samples, static_cast<int>(signal.size(0)) - min_trim);
    int num_windows = num_samples / window_size;

    for (int pos = 0; pos < num_windows; pos++) {
        int start = pos * window_size + min_trim;
        int end = start + window_size;

        auto window = signal.index({torch::indexing::Slice(start, end)});
        auto elements = window > threshold;

        if ((elements.sum().item<int>() > min_elements) || seen_peak) {
            seen_peak = true;
            if (window[-1].item<float>() > threshold) {
                continue;
            }
            if (end >= num_samples || end >= (max_trim * signal.size(0))) {
                return min_trim;
            } else {
                return end;
            }
        }
    }

    return min_trim;
}

}  // namespace dorado
