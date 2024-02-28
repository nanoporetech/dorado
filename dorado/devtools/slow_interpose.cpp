/**
 * Helpers to interpose threading primitive methods in order to slow down
 * execution and help TSan find issues that may otherwise be missed.
 */

#include "slow_interpose.h"

#include <dlfcn.h>

#include <atomic>
#include <cassert>
#include <thread>

#define CONCAT(a, b) a##b
#define STRINGIFY(x) #x
#define REAL_FUNC(name) CONCAT(__real_, name)

// Torch does some stuff during static init and destruction that seems to
// lock up if we sleep, so allow scoping of when to perform additional stuff.
namespace {
std::atomic_bool s_static_init_finished;
bool should_interpose() { return s_static_init_finished.load(std::memory_order_relaxed); }
}  // namespace

namespace dorado::slow_interpose {
ScopedSlowInterpose::ScopedSlowInterpose() {
    m_orig = s_static_init_finished.exchange(true, std::memory_order_relaxed);
}
ScopedSlowInterpose::~ScopedSlowInterpose() {
    s_static_init_finished.store(m_orig, std::memory_order_relaxed);
}
}  // namespace dorado::slow_interpose

// GCC supports direct wrapping, but for clang we have to dlsym() ourselves.
#ifndef __clang__
// Note: this requires adding -Wl,--wrap=name to the linker flags.
#warning "Not tested yet on Linux"
#define WRAP_FUNC(name, ret, args)   \
    extern ret REAL_FUNC(name) args; \
    extern "C" ret name args

#else  // __clang__
namespace {

auto* load_stdlib() {
#if defined(__APPLE__) && defined(_LIBCPP_VERSION) && _LIBCPP_ABI_VERSION == 1
    const char stdlib_path[] = "/usr/lib/libc++.1.dylib";
#else
#error "Add your platform's stdlib here"
#endif
    static auto* handle = dlopen(stdlib_path, RTLD_LAZY);
    assert(handle != nullptr);
    return handle;
}

template <typename Func, const char* m_name>
struct SymbolLoader {
    // Note: we're relying on static storage to set this to nullptr, since our
    // static init may come after we've already been called.
    std::atomic<Func> m_func;

    void load_symbol() {
        auto* real_func = dlsym(load_stdlib(), m_name);
        assert(real_func != nullptr);
        m_func.store(reinterpret_cast<Func>(real_func), std::memory_order_relaxed);
    }

    template <typename... Args>
    decltype(auto) operator()(Args&&... args) {
        auto* func = m_func.load(std::memory_order_relaxed);
        if (func == nullptr) {
            load_symbol();
            func = m_func.load(std::memory_order_relaxed);
        }
        return func(std::forward<Args>(args)...);
    }
};

}  // namespace

#define WRAP_FUNC(name, ret, args)                                        \
    static const char CONCAT(name, _str)[] = STRINGIFY(name);             \
    static SymbolLoader<ret(*) args, CONCAT(name, _str)> REAL_FUNC(name); \
    extern "C" ret name args

#endif  // __clang__

// Mangled function symbols.
#if defined(_LIBCPP_VERSION) && _LIBCPP_ABI_VERSION == 1
#define std_condition_variable_notify_one _ZNSt3__118condition_variable10notify_oneEv
#define std_condition_variable_notify_all _ZNSt3__118condition_variable10notify_allEv
#define std_mutex_unlock _ZNSt3__15mutex6unlockEv
#elif defined(__GLIBCXX__)  // symbols are the same for !_GLIBCXX_USE_CXX11_ABI
#define std_condition_variable_notify_one _ZNSt18condition_variable10notify_oneEv
#define std_condition_variable_notify_all _ZNSt18condition_variable10notify_allEv
#define std_mutex_unlock _ZNSt5mutex6unlockEv
#else
#error "Add your platform's symbols here"
#endif

WRAP_FUNC(std_condition_variable_notify_one, void, (void* self)) {
    REAL_FUNC(std_condition_variable_notify_one)(self);
    if (should_interpose()) {
        // It's legal to provide spurious wakeups, so wait a bit and poke it again.
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        REAL_FUNC(std_condition_variable_notify_one)(self);
    }
}

WRAP_FUNC(std_condition_variable_notify_all, void, (void* self)) {
    REAL_FUNC(std_condition_variable_notify_all)(self);
    if (should_interpose()) {
        // It's legal to provide spurious wakeups, so wait a bit and poke it again.
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        REAL_FUNC(std_condition_variable_notify_all)(self);
    }
}

WRAP_FUNC(std_mutex_unlock, void, (void* self)) {
    REAL_FUNC(std_mutex_unlock)(self);
    if (should_interpose()) {
        // Don't immediately start executing what comes after an unlock.
        // This REALLY slows down the tests, so don't sleep for long.
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}
