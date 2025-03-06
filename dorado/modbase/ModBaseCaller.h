#pragma once

#include "MotifMatcher.h"
#include "config/ModBaseModelConfig.h"
#include "torch_utils/module_utils.h"
#include "utils/stats.h"
#if DORADO_CUDA_BUILD
#include <c10/cuda/CUDAStream.h>
#endif
#include <torch/nn.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace dorado::modbase {

class ModBaseScaler;

class ModBaseCaller {
    struct ModBaseTask;

public:
    class ModBaseData {
        friend class ModBaseCaller;

    public:
        ModBaseData(const config::ModBaseModelConfig& config,
                    const at::TensorOptions& opts,
                    const int batch_size_);
        std::vector<size_t> get_motif_hits(const std::string& seq) const;
        int64_t get_sig_len() const;
        int64_t get_seq_len() const;

        const config::ModBaseModelConfig params;
        const std::vector<float> kmer_refinement_levels;
        std::unique_ptr<ModBaseScaler> scaler;

    private:
        utils::ModuleWrapper module_holder;
        const MotifMatcher matcher;
        std::deque<std::shared_ptr<ModBaseTask>> input_queue;
        std::mutex input_lock;
        std::condition_variable input_cv;
        const int batch_size;
#if DORADO_CUDA_BUILD
        c10::optional<c10::Stream> stream;
#endif
    };

    ModBaseCaller(const std::vector<std::filesystem::path>& model_paths,
                  const int batch_size,
                  const std::string& device);
    ~ModBaseCaller();

    std::vector<at::Tensor> create_input_sig_tensors() const;
    std::vector<at::Tensor> create_input_seq_tensors() const;
    c10::Device device() const { return m_options.device(); }

    at::Tensor call_chunks(size_t model_id,
                           at::Tensor& input_sigs,
                           at::Tensor& input_seqs,
                           int num_chunks);

    void terminate();
    void restart();

    std::string get_name() const {
        return std::string("ModBaseCaller_") + m_options.device().str();
    }

    stats::NamedStats sample_stats() const;

    const std::unique_ptr<ModBaseData>& modbase_model_data(size_t model_idx) const {
        return m_model_data.at(model_idx);
    }
    size_t num_models() const { return m_model_data.size(); }

private:
    void start_threads();
    void modbase_task_thread_fn(size_t model_id);

    const size_t m_num_models;

    at::TensorOptions m_options;
    std::atomic<bool> m_terminate{false};
    std::vector<std::unique_ptr<ModBaseData>> m_model_data;
    std::vector<std::thread> m_task_threads;

    // Performance monitoring stats.
    std::atomic<int64_t> m_num_batches_called = 0;
    std::atomic<int64_t> m_model_ms = 0;
};

}  // namespace dorado::modbase
