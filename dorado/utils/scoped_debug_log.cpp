#include "scoped_debug_log.h"

#include <spdlog/spdlog.h>

namespace dorado::utils {

namespace {
static const std::string SCOPED_LOG_ENTER{"[ENTRY]"};
static const std::string SCOPED_LOG_EXIT{"[EXIT]"};
static const std::string SCOPED_LOG_MESSAGE{"[MSG]"};
}  // namespace

ScopedDebugLog::ScopedDebugLog(std::string context) : m_context(std::move(context)) {
    write_log_message(SCOPED_LOG_ENTER, {});
}

ScopedDebugLog::~ScopedDebugLog() { write_log_message(SCOPED_LOG_EXIT, {}); }

void ScopedDebugLog::write_log_message(const std::string &message_type,
                                       const std::string &message) {
    spdlog::debug("{} {} {}", m_context, message_type, message);
}

void ScopedDebugLog::write(const std::string &message) {
    write_log_message(SCOPED_LOG_MESSAGE, message);
}

}  // namespace dorado::utils