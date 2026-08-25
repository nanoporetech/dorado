#pragma once

// Backport of std::move_only_function.

#include <functional>
#include <new>
#include <type_traits>
#include <utility>

// NOLINTBEGIN(cppcoreguidelines-pro-type-union-access): union accesses here are safe.

namespace dorado::utils {

namespace detail {

// Arbitrarily chosen small functor size.
static inline constexpr std::size_t kSmallFunctorAlignment = alignof(std::size_t);
static inline constexpr std::size_t kSmallFunctorSize = 3 * sizeof(std::size_t);

template <typename Func>
struct is_small {
    using F = std::remove_reference_t<Func>;
    static constexpr inline bool value =
            sizeof(F) <= kSmallFunctorSize && alignof(F) <= kSmallFunctorAlignment;
};

template <typename Func>
static inline constexpr bool is_small_v = is_small<Func>::value;

// Heap allocated callable object.
template <typename R, typename... Args>
struct ICallable {
    virtual ~ICallable() = default;
    virtual R invoke(Args&&...) = 0;
};

template <bool IsConst, typename R, typename... Args>
class MoveOnlyFunctionBase {
    using ICallable = detail::ICallable<R, Args...>;
    using ICallablePtr = ICallable*;
    using FunctionPtr = R (*)(Args...);

    // We either hold an inlined "small" functor, a heap allocated callable, or a function pointer.
    // Lifetime's are ensured by m_move_and_destroy().
    union Data {
        alignas(detail::kSmallFunctorAlignment) std::byte storage[detail::kSmallFunctorSize];
        ICallablePtr callable;
        FunctionPtr func_ptr;
    };
    Data m_data;

    // Move |src| into |dst| and destroy |src|. |src| should only be written to
    // after calling this since it'll be bogus.
    void (*m_move_and_destroy)(Data& src, Data* dst) = nullptr;

    // |m_invoke| is set iff we can be invoked.
    using InvokeData = std::conditional_t<IsConst, const Data, Data>;
    R (*m_invoke)(InvokeData& data, Args&&... args) = nullptr;

public:
    // Construct from a function pointer.
    template <typename Func>
        requires(std::is_invocable_r_v<R, Func, Args...> &&
                 std::is_convertible_v<Func, FunctionPtr>)
    MoveOnlyFunctionBase(Func func) noexcept : MoveOnlyFunctionBase() {
        FunctionPtr ptr = func;
        // Silence warning that |ptr| will always be non-NULL if |func| is the
        // address of a function.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waddress"
#endif
        if (ptr) {
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
            m_data.func_ptr = ptr;
            m_move_and_destroy = [](Data& src, Data* dst) {
                if (dst) {
                    dst->func_ptr = src.func_ptr;
                }
            };
            m_invoke = [](InvokeData& data, Args&&... args) {
                return data.func_ptr(std::forward<Args>(args)...);
            };
        }
    }

    // Construct from a big functor.
    template <typename Func>
        requires(std::is_invocable_r_v<R, Func, Args...> &&
                 !std::is_convertible_v<Func, FunctionPtr> && !detail::is_small_v<Func>)
    MoveOnlyFunctionBase(Func&& func) {
        using F = std::remove_reference_t<Func>;

        struct Callable : ICallable {
            F m_func;
            explicit Callable(Func&& f) : m_func(std::move(f)) {}
            R invoke(Args&&... args) override { return m_func(std::forward<Args>(args)...); }
        };

        m_data.callable = new Callable(std::forward<Func>(func));
        m_move_and_destroy = [](Data& src, Data* dst) {
            if (dst) {
                dst->callable = src.callable;
            } else {
                delete src.callable;
            }
        };
        m_invoke = [](InvokeData& data, Args&&... args) {
            return data.callable->invoke(std::forward<Args>(args)...);
        };
    }

    // Construct from a small functor.
    template <typename Func>
        requires(std::is_invocable_r_v<R, Func, Args...> &&
                 !std::is_convertible_v<Func, FunctionPtr> && detail::is_small_v<Func>)
    MoveOnlyFunctionBase(Func&& func) {
        using F = std::remove_reference_t<Func>;

        new (&m_data.storage) F(std::forward<Func>(func));
        m_move_and_destroy = [](Data& src, Data* dst) {
            F* from = std::launder(reinterpret_cast<F*>(src.storage));
            if (dst) {
                new (&dst->storage) F(std::move(*from));
            }
            from->~F();
        };
        m_invoke = [](InvokeData& data, Args&&... args) {
            using MaybeConstF = std::conditional_t<IsConst, const F, F>;
            auto* f = std::launder(reinterpret_cast<MaybeConstF*>(data.storage));
            return (*f)(std::forward<Args>(args)...);
        };
    }

    MoveOnlyFunctionBase(std::nullptr_t) noexcept : MoveOnlyFunctionBase() {}

    MoveOnlyFunctionBase() noexcept = default;
    MoveOnlyFunctionBase(MoveOnlyFunctionBase&) = delete;
    MoveOnlyFunctionBase& operator=(MoveOnlyFunctionBase&) = delete;
    MoveOnlyFunctionBase(MoveOnlyFunctionBase&& o) noexcept : MoveOnlyFunctionBase() {
        operator=(std::move(o));
    }
    MoveOnlyFunctionBase& operator=(MoveOnlyFunctionBase&& o) noexcept {
        if (&o != this) {
            // Destroy what we're holding.
            if (m_move_and_destroy) {
                m_move_and_destroy(m_data, nullptr);
            }

            // Move the contents of their storage into ours.
            if (o.m_move_and_destroy) {
                o.m_move_and_destroy(o.m_data, &m_data);
            }

            // Take ownership of and clear their vtable.
            m_move_and_destroy = std::exchange(o.m_move_and_destroy, nullptr);
            m_invoke = std::exchange(o.m_invoke, nullptr);
        }
        return *this;
    }

    ~MoveOnlyFunctionBase() {
        // Destroy what we're holding.
        if (m_move_and_destroy) {
            m_move_and_destroy(m_data, nullptr);
        }
    }

    R operator()(Args&&... args)
        requires(!IsConst)
    {
        if (!*this) {
            throw std::bad_function_call();
        }
        return m_invoke(m_data, std::forward<Args>(args)...);
    }
    R operator()(Args&&... args) const
        requires(IsConst)
    {
        if (!*this) {
            throw std::bad_function_call();
        }
        return m_invoke(m_data, std::forward<Args>(args)...);
    }

    explicit operator bool() const { return m_invoke != nullptr; }
};

}  // namespace detail

template <typename T>
class MoveOnlyFunction;

template <typename R, typename... Args>
class MoveOnlyFunction<R(Args...)> : public detail::MoveOnlyFunctionBase<false, R, Args...> {
    using Base = detail::MoveOnlyFunctionBase<false, R, Args...>;

public:
    using Base::Base;
};

template <typename R, typename... Args>
class MoveOnlyFunction<R(Args...) const> : public detail::MoveOnlyFunctionBase<true, R, Args...> {
    using Base = detail::MoveOnlyFunctionBase<true, R, Args...>;

public:
    using Base::Base;
};

}  // namespace dorado::utils

// NOLINTEND(cppcoreguidelines-pro-type-union-access)
