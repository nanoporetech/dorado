#include "alignment/Minimap2MemoryLimiter.h"

#include "utils/ResourceLimiter.h"

#include <minimap.h>

namespace dorado::alignment {

bool install_mm2_limiter_hooks(std::size_t max_memory_limit_GB, std::size_t num_threads) {
    // If something's already installed then we don't want to replace it.
    if (dorado_mm2_reserve || dorado_mm2_release) {
        return false;
    }

    // No max memory usage is probably a bug on the caller.
    if (max_memory_limit_GB == 0) {
        return false;
    }

    // We split the memory limit into big and small queues because the larger
    // the allocation the longer it takes to process and that can lead to an
    // entire pipeline stall for several minutes.
    //
    // Empirical testing on a random dataset suggests that the majority of
    // allocations are less than 5MB, so use that as the split point.
    //
    static constexpr std::size_t split_point = 5 * 1024 * 1024;

    // The small queue should be able to process 1 element per alignment worker,
    // which is going to be a few hundred MB at most. The max memory limit will
    // be multiple GB so this should be safe.
    const std::size_t small_queue_size = split_point * num_threads;
    const std::size_t total_queue_size = max_memory_limit_GB * 1024 * 1024 * 1024;
    if (total_queue_size <= small_queue_size) {
        return false;
    }

    // Create the limiters and our per-thread waiters.
    static utils::ResourceLimiter big_queue(total_queue_size - small_queue_size);
    static utils::ResourceLimiter small_queue(small_queue_size);
    thread_local utils::ResourceLimiter::WaiterState waiter;

    static constexpr auto queue_for_size = [](std::size_t size) -> utils::ResourceLimiter & {
        if (size > split_point) {
            return big_queue;
        } else {
            return small_queue;
        }
    };

    dorado_mm2_reserve = [](int64_t n_a) {
        // Minimap can call acquire() twice if it rechains, but it'll have freed
        // the original allocation so we should do that too.
        dorado_mm2_release();

        // There's 5 allocations for each element of |a|.
        const size_t size = n_a * (sizeof(mm128_t) +                       // collect_seed_hits()
                                   sizeof(int64_t) + 3 * sizeof(uint32_t)  // mg_lchain_dp()
                                  );

        queue_for_size(size).acquire(waiter, size);
    };
    dorado_mm2_release = []() {
        const std::size_t size = waiter.reserved;
        queue_for_size(size).release(waiter);
    };
    return true;
}

}  // namespace dorado::alignment
