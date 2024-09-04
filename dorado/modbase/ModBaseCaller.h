#pragma once

#include "ModBaseModelConfig.h"
#include "MotifMatcher.h"
#include "utils/module_utils.h"
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
        ModBaseData(const ModBaseModelConfig& config,
                    const at::TensorOptions& opts,
                    const int batch_size_);
        std::vector<size_t> get_motif_hits(const std::string& seq) const;

        const ModBaseModelConfig params;
        std::unique_ptr<ModBaseScaler> scaler;

    private:
        torch::nn::ModuleHolder<torch::nn::AnyModule> module_holder;
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
                  int batch_size,
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

    const std::unique_ptr<ModBaseData>& caller_data(size_t caller_id) {
        return m_caller_data[caller_id];
    }
    size_t num_model_callers() const { return m_caller_data.size(); }

private:
    void start_threads();
    void modbase_task_thread_fn(size_t model_id);

    const size_t m_num_models;

    at::TensorOptions m_options;
    std::atomic<bool> m_terminate{false};
    std::vector<std::unique_ptr<ModBaseData>> m_caller_data;
    std::vector<std::thread> m_task_threads;

    // Performance monitoring stats.
    std::atomic<int64_t> m_num_batches_called = 0;
    std::atomic<int64_t> m_model_ms = 0;
};

}  // namespace dorado::modbase
