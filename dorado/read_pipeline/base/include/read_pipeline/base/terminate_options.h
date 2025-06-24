#pragma once

namespace dorado {

struct TerminateOptions {
    // Terminate fast instead of processing all remaining messages.
    bool fast = false;
};

inline TerminateOptions DefaultTerminateOptions() { return {}; }

}  // namespace dorado
