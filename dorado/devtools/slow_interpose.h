#pragma once

namespace dorado::slow_interpose {
// Scoped wrapper to be added around the code that you want to thoroughly
// check with TSan.
class [[nodiscard]] ScopedSlowInterpose {
    bool m_orig;
    ScopedSlowInterpose(const ScopedSlowInterpose&) = delete;
    ScopedSlowInterpose& operator=(const ScopedSlowInterpose&) = delete;

public:
    ScopedSlowInterpose();
    ~ScopedSlowInterpose();
};
}  // namespace dorado::slow_interpose
