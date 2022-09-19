#include "ScalerNode.h"

#include <algorithm>
#include <chrono>

using namespace std::chrono_literals;

std::pair<float, float> quantile(const torch::Tensor t) {
    // fast q20 and q90 using nth_element
    auto tmp = t.clone();
    auto start = tmp.data_ptr<float>();
    auto end = tmp.data_ptr<float>() + tmp.size(0);

    auto t1 = tmp.data_ptr<float>() + int(tmp.size(0) * 0.2);
    std::nth_element(start, t1, end);
    auto v1 = tmp[int(tmp.size(0) * 0.2)].item<float>();

    auto t2 = tmp.data_ptr<float>() + int(tmp.size(0) * 0.9);
    std::nth_element(start, t2, end);
    auto v2 = tmp[int(tmp.size(0) * 0.9)].item<float>();

    return std::make_pair(v1, v2);
}

std::pair<float, float> normalisation(torch::Tensor& x) {
    //Calculate shift and scale factors for normalisation.
    //auto quantiles = torch::quantile(x.index({torch::indexing::Slice(0, 8000)}),
    //                                 torch::tensor({0.2, 0.9}, {torch::kFloat}));

    //float q20 = quantiles[0].item<float>();
    //float q90 = quantiles[1].item<float>();

    auto [q20, q90] = quantile(x);

    float shift = std::max(10.0f, 0.51f * (q20 + q90));
    float scale = std::max(1.0f, 0.53f * (q90 - q20));

    return std::make_pair(shift, scale);
}

void ScalerNode::worker_thread() {
    while (true) {
        // Wait until we are provided with a read
        std::unique_lock<std::mutex> lock(m_cv_mutex);
        m_cv.wait_for(lock, 100ms, [this] { return !m_reads.empty(); });
        if (m_reads.empty()) {
            if (m_terminate) {
                // Notify our sink and then kill the worker if we're done
                m_sink.terminate();
                return;
            } else {
                continue;
            }
        }

        std::shared_ptr<Read> read = m_reads.front();
        m_reads.pop_front();
        lock.unlock();

        if (!read->scale_set) {
            read->scaling = (float)read->range / (float)read->digitisation;
            read->scale_set = true;
        }

        read->raw_data = read->scaling * (read->raw_data + read->offset);

        auto [shift, scale] = normalisation(read->raw_data);
        read->shift = shift;
        read->scale = scale;
        read->raw_data = (read->raw_data - read->shift) / read->scale;

        float threshold = shift + scale * 2.4;

        //8000 value may be changed in future. Currently this is found to work well.
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

ScalerNode::ScalerNode(ReadSink& sink, size_t max_reads)
        : ReadSink(max_reads),
          m_sink(sink),
          m_worker(new std::thread(&ScalerNode::worker_thread, this)) {}

ScalerNode::~ScalerNode() {
    terminate();
    m_cv.notify_one();
    m_worker->join();
}

int ScalerNode::trim(torch::Tensor signal,
                     int window_size,
                     float threshold,
                     int min_elements,
                     int max_samples,
                     float max_trim) {
    int min_trim = 10;
    bool seen_peak = false;
    int num_samples = std::min(max_samples, (int)signal.size(0));
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
            if (end >= num_samples || end / (int)signal.size(0) > max_trim) {
                return min_trim;
            } else {
                return end;
            }
        }
    }

    return min_trim;
}
