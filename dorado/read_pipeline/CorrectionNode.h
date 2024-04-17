#pragma once

#include "read_pipeline/MessageSink.h"
#include "read_pipeline/messages.h"
#include "utils/AsyncQueue.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <torch/nn/utils/rnn.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace dorado {

struct OverlapWindow {
    size_t overlap_idx = -1;
    int tstart = -1;
    int qstart = -1;
    int qend = -1;
    int cigar_start_idx = -1;
    int cigar_start_offset = -1;
    int cigar_end_idx = -1;
    int cigar_end_offset = -1;
    float accuracy = 0;
};

struct WindowFeatures {
    torch::Tensor bases;
    torch::Tensor quals;
    torch::Tensor indices;
    int length;
    std::vector<std::pair<int, int>> supported;
    std::vector<char> inferred_bases;
    int n_alns = 0;
    std::string read_name = "";
    int window_idx = -1;
};

struct base_count_t {
    int c = 0;
    char b;
};

class CorrectionNode : public MessageSink {
public:
    CorrectionNode(int threads, int bach_size);
    ~CorrectionNode() { stop_input_processing(); }
    std::string get_name() const override { return "CorrectionNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions&);  // override { stop_input_processing(); }
    void restart() override { start_input_processing(&CorrectionNode::input_thread_fn, this); }

private:
    void input_thread_fn();
    const int m_window_size = 4096;
    int m_batch_size;

    bool filter_overlap(const OverlapWindow& overlap, const CorrectionAlignments& alignments);
    void calculate_accuracy(OverlapWindow& overlap,
                            const CorrectionAlignments& alignments,
                            size_t win_idx,
                            int win_len);
    std::vector<int> get_max_ins_for_window(const std::vector<OverlapWindow>& overlaps,
                                            const CorrectionAlignments& alignments,
                                            int tstart,
                                            int win_len);
    std::tuple<torch::Tensor, torch::Tensor> get_features_for_window(
            const std::vector<OverlapWindow>& overlaps,
            const CorrectionAlignments& alignments,
            int win_len,
            int tstart,
            const std::vector<int>& max_ins);
    std::vector<std::pair<int, int>> get_supported(torch::Tensor& bases);
    torch::Tensor get_indices(torch::Tensor bases, std::vector<std::pair<int, int>>& supported);
    std::vector<WindowFeatures> extract_features(std::vector<std::vector<OverlapWindow>>& windows,
                                                 const CorrectionAlignments& alignments);
    void extract_windows(std::vector<std::vector<OverlapWindow>>& windows,
                         const CorrectionAlignments& alignments);
    std::vector<std::string> decode_windows(const std::vector<WindowFeatures>& wfs);

    void infer_fn();
    void decode_fn();

    utils::AsyncQueue<WindowFeatures> m_features_queue;
    utils::AsyncQueue<WindowFeatures> m_inferred_features_queue;

    std::unique_ptr<std::thread> m_infer_thread;
    std::vector<std::unique_ptr<std::thread>> m_decode_threads;

    std::chrono::duration<double> extractWindowsDuration;
    std::mutex ewMutex;
    std::chrono::duration<double> extractFeaturesDuration;
    std::mutex efMutex;
    std::chrono::duration<double> runInferenceDuration;
    std::mutex riMutex;
    std::chrono::duration<double> decodeDuration;
    std::mutex decodeMutex;
    std::atomic<int> num_reads;

    std::unordered_map<std::string, std::vector<WindowFeatures>> m_features_by_id;
    std::unordered_map<std::string, int> m_pending_features_by_id;
    std::mutex m_features_mutex;

    std::atomic<int> m_num_active_feature_threads{0};
    std::atomic<int> m_num_active_infer_threads{0};
    std::atomic<int> m_num_active_decode_threads{0};
};

}  // namespace dorado
