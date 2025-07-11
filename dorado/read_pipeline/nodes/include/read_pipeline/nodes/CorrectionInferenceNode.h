#pragma once

#include "correct/types.h"
#include "read_pipeline/base/MessageSink.h"

#include <spdlog/spdlog.h>

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace dorado {

class CorrectionInferenceNode : public MessageSink {
public:
    CorrectionInferenceNode(const std::string& fastq,
                            int threads,
                            const std::string& device,
                            int infer_threads,
                            int bach_size,
                            const std::filesystem::path& model_dir,
                            const bool legacy_windowing,
                            const std::unordered_set<std::string>& debug_tnames);
    ~CorrectionInferenceNode();

    std::string get_name() const override;
    stats::NamedStats sample_stats() const override;
    void terminate(const TerminateOptions&) override;
    void restart() override;

private:
    const std::string m_fastq;
    const correction::ModelConfig m_model_config;
    void input_thread_fn();

    void terminate_impl(utils::AsyncQueueTerminateFast fast);

    void infer_fn(const std::string& device, int mtx_idx, int batch_size);
    void decode_fn();

    void concat_features_and_send(const std::vector<std::string>& seqs,
                                  const std::string& read_name);

    utils::AsyncQueue<correction::WindowFeatures> m_features_queue;
    utils::AsyncQueue<correction::WindowFeatures> m_inferred_features_queue;

    std::vector<std::thread> m_infer_threads;
    std::vector<std::thread> m_decode_threads;

    std::atomic<int> num_reads{0};
    std::atomic<int> num_early_reads{0};
    std::atomic<int> total_reads_in_input{0};

    std::unordered_map<std::string, std::vector<std::string>> m_features_by_id;
    std::unordered_map<std::string, int> m_pending_features_by_id;
    std::mutex m_features_mutex;

    std::atomic<int> m_num_active_feature_threads{0};

    std::array<std::mutex, 32> m_gpu_mutexes;

    bool m_legacy_windowing = false;
    std::unordered_set<std::string> m_debug_tnames;
};

}  // namespace dorado
