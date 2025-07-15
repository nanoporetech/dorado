#pragma once

#include "utils/AsyncQueue.h"

namespace dorado {

struct TerminateOptions {
    // Whether or not to preserve the PairingNode's internal cache instead of flushing it.
    bool preserve_pairing_caches = false;

    // Terminate fast instead of processing all remaining messages.
    utils::AsyncQueueTerminateFast fast = utils::AsyncQueueTerminateFast::No;
};

}  // namespace dorado
