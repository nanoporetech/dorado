#pragma once

namespace dorado::utils {

// Initialises the default logger to point to stderr.
void InitLogging();

enum class VerboseLogLevel {
    DEBUG = 1,
    TRACE = 2,
};

void SetVerboseLogging(VerboseLogLevel level);

}  // namespace dorado::utils
