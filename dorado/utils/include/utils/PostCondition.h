#pragma once

#include <utility>

namespace dorado::utils {

namespace detail {
template <typename Func>
class PostCondition {
public:
    explicit PostCondition(Func&& func) : m_func(std::move(func)) {}
    ~PostCondition() { m_func(); }

private:
    PostCondition(const PostCondition&) = delete;
    PostCondition& operator=(const PostCondition&) = delete;
    PostCondition(PostCondition&&) = delete;
    PostCondition& operator=(PostCondition&&) = delete;

private:
    Func m_func;
};
}  // namespace detail

template <typename Func>
[[nodiscard]] auto PostCondition(Func function) {
    return detail::PostCondition(std::move(function));
}

}  // namespace dorado::utils
