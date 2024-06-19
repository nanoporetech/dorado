#pragma once

namespace dorado {

class ThreadNaming {
public:
    virtual ~ThreadNaming() = default;
    virtual void set_thread_name(const std::string& name) = 0;
};

}  // namespace dorado