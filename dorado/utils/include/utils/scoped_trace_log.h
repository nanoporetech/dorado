#pragma once

#include <spdlog/spdlog.h>

#define DORADO_TRACE_SCOPE(scope) ::dorado::utils::detail::ScopedTraceLog scope(__func__)
#define DORADO_TRACE_MSG(scope, fmt, ...)                                 \
    do {                                                                  \
        if (scope.enabled()) {                                            \
            spdlog::trace("{} [MSG] " fmt, scope.context(), __VA_ARGS__); \
        }                                                                 \
    } while (false)

namespace dorado::utils::detail {

// Writes entry/exit messages to trace logging with the given context name
class [[nodiscard]] ScopedTraceLog final {
    const char *m_context;
    bool m_enabled;

private:
    ScopedTraceLog(const ScopedTraceLog &) = delete;
    ScopedTraceLog &operator=(const ScopedTraceLog &) = delete;
    ScopedTraceLog(ScopedTraceLog &&) = delete;
    ScopedTraceLog &operator=(ScopedTraceLog &&) = delete;

public:
    explicit ScopedTraceLog(const char *context)
            : m_context(context), m_enabled(spdlog::should_log(spdlog::level::level_enum::trace)) {
        if (m_enabled) {
            spdlog::trace("{} [ENTRY]", m_context);
        }
    }
    ~ScopedTraceLog() {
        if (m_enabled) {
            spdlog::trace("{} [EXIT]", m_context);
        }
    }

    const char *context() const { return m_context; }
    bool enabled() const { return m_enabled; }
};

}  // namespace dorado::utils::detail
