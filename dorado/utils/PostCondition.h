#pragma once
#include <functional>

namespace dorado::utils {

class PostCondition {
public:
    PostCondition(std::function<void()> func) : m_func(func) {}
    ~PostCondition() {
        if (m_func) {
            m_func();
        }
    }

private:
    std::function<void()> m_func;
};

}  // namespace dorado::utils
