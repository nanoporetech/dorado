#pragma once

struct KoiThreadPool;

namespace dorado::nn {

// TODO: These should really be part of Koi
bool koi_can_use_cutlass(/* current device */);
bool koi_can_use_cutlass(int device_id);
bool koi_can_use_quantised_lstm(/* current device */);

// Helper wrapper around a KoiThreadPool.
class KoiThreads {
    KoiThreadPool *m_threads = nullptr;

    KoiThreads(const KoiThreads &) = delete;
    KoiThreads(KoiThreads &&) = delete;
    KoiThreads &operator=(const KoiThreads &) = delete;
    KoiThreads &operator=(KoiThreads &&) = delete;

public:
    explicit KoiThreads(int num_threads);
    ~KoiThreads();

    KoiThreadPool *get() { return m_threads; }
};

}  // namespace dorado::nn
