#pragma once

#include <string>

namespace dorado::utils {

using GetStacktrace = std::string();
void set_stacktrace_getter(GetStacktrace *getter);

// Install a handler that tries to log a message if we segfault.
void install_segfault_handler();

// Install a handler that tries to log a message on an uncaught exception.
void install_uncaught_exception_handler();

}  // namespace dorado::utils
