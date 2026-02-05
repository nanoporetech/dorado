#include "alignment/Minimap2MemoryLimiter.h"

#include "utils/ResourceLimiter.h"

#include <minimap.h>

namespace dorado::alignment {

bool install_mm2_limiter_hooks(std::size_t max_memory_limit) {
    // If something's already installed then we don't want to replace it.
    if (dorado_mm2_reserve || dorado_mm2_release) {
        return false;
    }

    // No max memory usage is probably a bug on the caller.
    if (max_memory_limit == 0) {
        return false;
    }

    // Create the limiter and our per-thread waiters.
    static utils::ResourceLimiter limiter(max_memory_limit);
    thread_local utils::ResourceLimiter::WaiterState waiter;

    dorado_mm2_reserve = [](int64_t n_a) {
        // Minimap can call acquire() twice if it rechains, but it'll have freed
        // the original allocation so we should do that too.
        limiter.release(waiter);

        // There's 5 allocations for each element of |a|.
        const size_t size = n_a * (sizeof(mm128_t) +                       // collect_seed_hits()
                                   sizeof(int64_t) + 3 * sizeof(uint32_t)  // mg_lchain_dp()
                                  );
        limiter.acquire(waiter, size);
    };
    dorado_mm2_release = []() { limiter.release(waiter); };
    return true;
}

}  // namespace dorado::alignment
