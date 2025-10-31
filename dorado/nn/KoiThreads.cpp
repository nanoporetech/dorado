#include "nn/KoiThreads.h"

#include "koi_thread_pool.h"

#include <spdlog/spdlog.h>

#include <stdexcept>

namespace dorado::nn {

KoiThreads::KoiThreads(int num_threads) {
    const int ret = koi_create_thread_pool(&m_thread_pool, num_threads);
    if (ret != KOI_SUCCESS) {
        throw std::runtime_error("Failed to create koi thread pool");
    }
}

KoiThreads::~KoiThreads() {
    const int ret = koi_destroy_thread_pool(&m_thread_pool);
    if (ret != KOI_SUCCESS) {
        spdlog::error("Failed to destroy koi thread pool");
    }
}

}  // namespace dorado::nn
