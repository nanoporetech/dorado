#pragma once

namespace dorado {

// Declaration of the thread naming interface.
class ThreadNaming {
public:
    virtual ~ThreadNaming() = default;

    // Call to set the name of the current thread.
    // N.B. the name will be truncated to 15 characters on some platforms.
    virtual void set_thread_name(const std::string& name) = 0;
};

}  // namespace dorado