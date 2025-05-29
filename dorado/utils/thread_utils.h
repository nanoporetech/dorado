#pragma once

#include <string>

namespace dorado::utils {

// Call to set the name of the current thread.
// N.B. the name will be truncated to 15 characters on some platforms.
void set_thread_name(const char* name);

// Run some busy work to keep the CPU clocked up at full speed.
void start_busy_work();
void stop_busy_work();

}  // namespace dorado::utils
