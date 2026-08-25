#pragma once

// Backport of std::function_ref.

#include <functional>
#include <new>
#include <type_traits>
#include <utility>

namespace dorado::utils {

namespace detail {

template <bool IsConst, typename R, typename... Args>
class FunctionRefBase;

template <typename T>
struct is_function_ref : std::false_type {};
template <bool IsConst, typename R, typename... Args>
struct is_function_ref<FunctionRefBase<IsConst, R, Args...>> : std::true_type {};
template <typename T>
inline constexpr bool is_function_ref_v = is_function_ref<T>::value;

template <bool IsConst, typename R, typename... Args>
class FunctionRefBase {
    using FunctionPtr = R (*)(Args...);

    // |m_invoke| is set iff we can be invoked.
    const void* m_functor = nullptr;
    R (*m_invoke)(const void* functor, Args&&... args) = nullptr;

public:
    FunctionRefBase() noexcept = default;

    FunctionRefBase(std::nullptr_t) : FunctionRefBase() {}

    FunctionRefBase(FunctionPtr ptr) : FunctionRefBase() {
        if (ptr) {
            m_functor = reinterpret_cast<void*>(ptr);
            m_invoke = [](const void* functor, Args&&... args) {
                auto* f = reinterpret_cast<FunctionPtr>(const_cast<void*>(functor));
                return (*f)(std::forward<Args>(args)...);
            };
        }
    }

    template <typename Func>
        requires(!is_function_ref_v<Func>)
    FunctionRefBase(Func&& func) {
        using F = std::remove_reference_t<Func>;

        m_functor = std::addressof(func);
        m_invoke = [](const void* functor, Args&&... args) {
            auto* f = std::launder(reinterpret_cast<F*>(const_cast<void*>(functor)));
            return (*f)(std::forward<Args>(args)...);
        };
    }

    R operator()(Args&&... args)
        requires(!IsConst)
    {
        if (!*this) {
            throw std::bad_function_call();
        }
        return m_invoke(m_functor, std::forward<Args>(args)...);
    }
    R operator()(Args&&... args) const
        requires(IsConst)
    {
        if (!*this) {
            throw std::bad_function_call();
        }
        return m_invoke(m_functor, std::forward<Args>(args)...);
    }

    explicit operator bool() const { return m_invoke != nullptr; }
};

}  // namespace detail

template <typename T>
class FunctionRef;

template <typename R, typename... Args>
class FunctionRef<R(Args...)> : public detail::FunctionRefBase<false, R, Args...> {
    using Base = detail::FunctionRefBase<false, R, Args...>;

public:
    using Base::Base;
};

template <typename R, typename... Args>
class FunctionRef<R(Args...) const> : public detail::FunctionRefBase<true, R, Args...> {
    using Base = detail::FunctionRefBase<true, R, Args...>;

public:
    using Base::Base;
};

}  // namespace dorado::utils
