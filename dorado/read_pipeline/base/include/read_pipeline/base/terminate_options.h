#pragma once

#include "utils/AsyncQueue.h"

namespace dorado {

struct TerminateOptions {
    // Terminate fast instead of processing all remaining messages.
    utils::AsyncQueueTerminateFast fast = utils::AsyncQueueTerminateFast::No;
};

}  // namespace dorado
