#pragma once

#include <string>

namespace dorado::utils {

// Writes entry/exit messages to debug logging with the given context name
class [[nodiscard]] ScopedDebugLog final {
    const std::string m_context;

    void write_log_message(const std::string &message_type, const std::string &message);

public:
    ScopedDebugLog(std::string context);
    ~ScopedDebugLog();

    // Writes a message in the context of the scoped log
    void write(const std::string &message);
};

}  // namespace dorado::utils