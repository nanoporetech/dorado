#pragma once

struct KoiThreadPool;

namespace dorado::nn {

// Helper wrapper around a KoiThreadPool.
class KoiThreads {
    KoiThreadPool *m_thread_pool = nullptr;

    KoiThreads(const KoiThreads &) = delete;
    KoiThreads(KoiThreads &&) = delete;
    KoiThreads &operator=(const KoiThreads &) = delete;
    KoiThreads &operator=(KoiThreads &&) = delete;

public:
    explicit KoiThreads(int num_threads);
    ~KoiThreads();

    KoiThreadPool *get() { return m_thread_pool; }
};

}  // namespace dorado::nn
