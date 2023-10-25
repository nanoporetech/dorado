#pragma once

namespace dorado::utils {

// Initialises the default logger to point to stderr.
void InitLogging();

enum DebugLogLevel {
    DEBUG = 0,
    TRACE,
};

void SetDebugLogging(DebugLogLevel level = DEBUG);

}  // namespace dorado::utils
