#pragma once

#include "utils/dev_utils.h"

#include <nvtx3/nvtx3.hpp>

// Set this to >0 to enable output of GPU profiling information to stderr
// or use `dorado [basecaller|duplex] ... --devopts cuda_profile_level=<X> ...`
#define CUDA_PROFILE_LEVEL_DEFAULT 0

#if DORADO_CUDA_BUILD
#include "cuda_utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#elif DORADO_METAL_BUILD
#include "metal_utils.h"

namespace dorado::utils::detail {
inline os_log_t s_scoped_os_log =
        os_log_create("scoped_profile", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
}  // namespace dorado::utils::detail

#endif

namespace dorado::utils {

// If `detail_level <= CUDA_PROFILE_LEVEL_DEFAULT`, this times a range and prints it to stderr so
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
        activate();
    }

    ~ScopedProfileRange() { finish(); }

private:
    ScopedProfileRange(const ScopedProfileRange &) = delete;
    ScopedProfileRange(ScopedProfileRange &&) = delete;
    ScopedProfileRange &operator=(const ScopedProfileRange &) = delete;
    ScopedProfileRange &operator=(ScopedProfileRange &&) = delete;

    void activate() {
        if (!m_active) {
            return;
        }
#if DORADO_CUDA_BUILD
        m_stream = at::cuda::getCurrentCUDAStream().stream();
        handle_cuda_result(cudaEventCreate(&m_start));
        handle_cuda_result(cudaEventRecord(m_start, m_stream));
#elif DORADO_METAL_BUILD
        m_signpost_id = os_signpost_id_generate(detail::s_scoped_os_log);
        os_signpost_interval_begin(detail::s_scoped_os_log, m_signpost_id, "ScopedProfileRange",
                                   "[%i]%s", m_detail_level, m_label);
#endif
    }

    void finish() {
        if (!m_active) {
            return;
        }
        m_active = false;

#if DORADO_CUDA_BUILD
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
#elif DORADO_METAL_BUILD
        os_signpost_interval_end(detail::s_scoped_os_log, m_signpost_id, "ScopedProfileRange",
                                 "[%i]%s", m_detail_level, m_label);
#endif
    }

    const char *m_label;
    int m_detail_level;
    bool m_active;

#if DORADO_CUDA_BUILD
    cudaStream_t m_stream;
    cudaEvent_t m_start;
#elif DORADO_METAL_BUILD
    os_signpost_id_t m_signpost_id = OS_SIGNPOST_ID_NULL;
#endif
};

}  // namespace dorado::utils
