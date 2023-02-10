#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

namespace dorado::utils {

void InitLogging() {
    // Without modification, the default logger will write to stdout.
    // Replace the default logger with a (color, multi-threaded) stderr logger
    // (but first replace it with an arbitrarily-named logger to prevent a name clash)
    spdlog::set_default_logger(spdlog::stderr_color_mt("unused_name"));
    spdlog::set_default_logger(spdlog::stderr_color_mt(""));
}

}  // namespace dorado::utils
