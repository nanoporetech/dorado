#pragma once

namespace dorado {

struct FlushOptions {
    bool preserve_pairing_caches = false;
};
inline FlushOptions DefaultFlushOptions() { return {false}; }

}  // namespace dorado