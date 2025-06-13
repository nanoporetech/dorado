#pragma once

#include <string>

namespace dorado::utils {

// Writes entry/exit messages to trace logging with the given context name
class [[nodiscard]] ScopedTraceLog final {
    const std::string m_context;

    void write_log_message(const std::string &message_type, const std::string &message);

public:
    ScopedTraceLog(std::string context);
    ~ScopedTraceLog();

    // Writes a message in the context of the scoped log
    void write(const std::string &message);
};

}  // namespace dorado::utils