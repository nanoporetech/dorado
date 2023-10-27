#pragma once

namespace dorado::utils {

// Initialises the default logger to point to stderr.
void InitLogging();

enum DebugLogLevel {
    DEBUG = 1,
    TRACE = 2,
};

void SetDebugLogging(DebugLogLevel level);

}  // namespace dorado::utils
