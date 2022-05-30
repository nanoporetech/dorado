#include "ScalerNode.h"

#include <chrono>
#define EPS 1e-9f

using namespace std::chrono_literals;

std::pair<float, float> calculate_med_mad(torch::Tensor& x, float factor = 1.4826) {
    //Calculate signal median and median absolute deviation
    auto med = x.median();
    auto mad = torch::median(torch::abs(x - med)) * factor + EPS;

    return {med.item<float>(), mad.item<float>()};
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
            read->scale = (float)read->range / (float)read->digitisation;
            read->scale_set = true;
        }

        read->raw_data = read->scale * (read->raw_data + read->offset);

        //8000 value may be changed in future. Currently this is found to work well.
        int trim_start =
                trim(read->raw_data.index({torch::indexing::Slice(torch::indexing::None, 8000)}));

        read->raw_data =
                read->raw_data.index({torch::indexing::Slice(trim_start, torch::indexing::None)});
        read->num_trimmed_samples = trim_start;

        auto med_mad = calculate_med_mad(read->raw_data);
        read->med = med_mad.first;
        read->mad = med_mad.second;

        read->raw_data = (read->raw_data - read->med) / std::max(1.0f, read->mad);

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
                     float threshold_factor,
                     int min_elements) {
    int min_trim = 10;
    signal = signal.index({torch::indexing::Slice(min_trim, torch::indexing::None)});

    auto trim_start = -(window_size * 100);

    auto trimmed = signal.index({torch::indexing::Slice(trim_start, torch::indexing::None)});
    auto med_mad = calculate_med_mad(trimmed);

    auto threshold = med_mad.first + med_mad.second * threshold_factor;

    auto signal_len = signal.size(0);
    int num_windows = signal_len / window_size;

    bool seen_peak = false;

    for (int pos = 0; pos < num_windows; pos++) {
        int start = pos * window_size;
        int end = start + window_size;

        auto window = signal.index({torch::indexing::Slice(start, end)});
        auto elements = window > threshold;

        if ((elements.sum().item<int>() > min_elements) || seen_peak) {
            seen_peak = true;
            if (window[-1].item<float>() > threshold) {
                continue;
            }
            return std::min(end + min_trim, (int)signal.size(0));
        }
    }

    return min_trim;
}
