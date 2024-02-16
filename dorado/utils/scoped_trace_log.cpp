#include "scoped_trace_log.h"

#include <spdlog/spdlog.h>

namespace dorado::utils {

namespace {
const std::string SCOPED_LOG_ENTER{"[ENTRY]"};
const std::string SCOPED_LOG_EXIT{"[EXIT]"};
const std::string SCOPED_LOG_MESSAGE{"[MSG]"};
}  // namespace

ScopedTraceLog::ScopedTraceLog(std::string context) : m_context(std::move(context)) {
    write_log_message(SCOPED_LOG_ENTER, {});
}

ScopedTraceLog::~ScopedTraceLog() { write_log_message(SCOPED_LOG_EXIT, {}); }

void ScopedTraceLog::write_log_message(const std::string &message_type,
                                       const std::string &message) {
    spdlog::trace("{} {} {}", m_context, message_type, message);
}

void ScopedTraceLog::write(const std::string &message) {
    write_log_message(SCOPED_LOG_MESSAGE, message);
}

}  // namespace dorado::utils