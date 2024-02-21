#pragma once

#include <utility>

namespace dorado::utils {

namespace detail {
template <typename Func>
class PostCondition {
public:
    PostCondition(Func&& func) : m_func(std::move(func)) {}
    ~PostCondition() { m_func(); }

private:
    Func m_func;
};
}  // namespace detail

template <typename Func>
[[nodiscard]] auto PostCondition(Func function) {
    return detail::PostCondition(std::move(function));
}

}  // namespace dorado::utils
