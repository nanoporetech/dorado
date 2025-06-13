#pragma once

// Set this to >0 to enable output of GPU profiling information to stderr
// or use `dorado [basecaller|duplex] ... --devopts cuda_profile_level=<X> ...`
#define CUDA_PROFILE_LEVEL_DEFAULT 0

#if DORADO_CUDA_BUILD
#include "cuda_utils.h"
#include "utils/dev_utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

namespace dorado::utils {
// If `detail_level <= CUDA_PROFILE_TO_CERR_LEVEL`, this times a range and prints it to stderr so
// you don't have to generate a QDREP file to perform basic profiling.
// Also is a nvtx3::scoped_range which means `label` will be shown in NSight tools.
class ScopedProfileRange : public nvtx3::scoped_range {
public:
    explicit ScopedProfileRange(const char *label, int detail_level)
            : nvtx3::scoped_range(label),
              m_label(label),
              m_detail_level(detail_level),
              m_active(m_detail_level <=
                       get_dev_opt<int>("cuda_profile_level", CUDA_PROFILE_LEVEL_DEFAULT)) {
        if (m_active) {
            m_stream = at::cuda::getCurrentCUDAStream().stream();
            handle_cuda_result(cudaEventCreate(&m_start));
            handle_cuda_result(cudaEventRecord(m_start, m_stream));
        }
    }

    ~ScopedProfileRange() { finish(); }

private:
    void finish() {
        if (!m_active) {
            return;
        }
        cudaEvent_t stop;
        handle_cuda_result(cudaEventCreate(&stop));
        handle_cuda_result(cudaEventRecord(stop, m_stream));
        handle_cuda_result(cudaEventSynchronize(stop));
        float timeMs = 0.0f;
        handle_cuda_result(cudaEventElapsedTime(&timeMs, m_start, stop));
        handle_cuda_result(cudaEventDestroy(m_start));
        handle_cuda_result(cudaEventDestroy(stop));
        std::cerr << std::string(m_detail_level - 1, '\t') << "[" << m_label << " " << timeMs
                  << " ms]\n";
        m_active = false;
    }

    const char *m_label;
    cudaStream_t m_stream;
    cudaEvent_t m_start;
    int m_detail_level;
    bool m_active;
};

}  // namespace dorado::utils

#else
namespace dorado::utils {
// Do nothing on Apple platforms
struct ScopedProfileRange {
    explicit ScopedProfileRange(const char *, int) {}
};
}  // namespace dorado::utils

#endif
