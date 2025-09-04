#include "utils/crash_handlers.h"

#include <spdlog/spdlog.h>

#include <atomic>
#include <csignal>
#include <exception>

#ifdef _WIN32
// Reduce the impact of Windows.h
#define NOGDI
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

namespace dorado::utils {

namespace {

std::atomic_bool s_enable_uncaught_exception_handler;
std::atomic<GetStacktrace *> s_get_stacktrace;

// One per thread since MSVC is non-conforming and doesn't have a single global termination handler.
#ifdef _WIN32
thread_local
#endif  // _WIN32
        void (*s_previous_terminate)();

std::string get_stacktrace() {
    // Until we have C++23's <stacktrace> we need to inject boost's or torch's implementation.
    GetStacktrace *getter = s_get_stacktrace.load(std::memory_order_acquire);
    if (!getter) {
        getter = [] {
            return std::string("Can't get trace due to no stacktrace() implementation");
        };
    }
    return getter();
}

struct GiveGoodTracesOnUncaughtExceptions {
    static void on_terminate() {
        if (s_enable_uncaught_exception_handler.load(std::memory_order_relaxed)) {
            // Print out where we are.
            spdlog::error("Uncaught exception thrown from:\n{}", get_stacktrace());

            // Print the exception that was thrown.
            auto current_exception = std::current_exception();
            if (current_exception != nullptr) {
                try {
                    std::rethrow_exception(current_exception);
                } catch (const std::exception &e) {
                    spdlog::error("Exception thrown: {}", e.what());
                } catch (...) {
                    spdlog::error("Unknown exception thrown");
                }
            } else {
                spdlog::error("No exception thrown");
            }
        }

        // And die.
        if (s_previous_terminate) {
            s_previous_terminate();
        } else {
            std::abort();
        }
    }

    GiveGoodTracesOnUncaughtExceptions() {
        // Not sure where set_terminate() lives on MSVC since it's non-conforming.
        using namespace std;
        s_previous_terminate = set_terminate(on_terminate);
    }
};

// One per thread since MSVC is non-conforming and doesn't have a single global termination handler/
#ifdef _WIN32
thread_local
#endif  // _WIN32
        GiveGoodTracesOnUncaughtExceptions s_widowmaker;

#ifndef _WIN32
template <sig_t *previous_handler>
void register_for_signal(int sig) {
    *previous_handler = signal(sig, [](int signum) {
        spdlog::error("Uncaught signal from:\n{}", get_stacktrace());
        // Reset to original handler and invoke it
        signal(signum, *previous_handler);
        raise(signum);
    });
}

sig_t s_sigsegv_handler, s_sigbus_handler, s_sigill_handler;
#endif

}  // namespace

void set_stacktrace_getter(GetStacktrace *getter) {
    s_get_stacktrace.store(getter, std::memory_order_release);
}

void install_segfault_handler() {
#ifdef _WIN32
    // Hook into SEH.
    static PTOP_LEVEL_EXCEPTION_FILTER const s_previous_handler = [] {
        return SetUnhandledExceptionFilter([](LPEXCEPTION_POINTERS exp) -> LONG {
            spdlog::error("Uncaught structured exception from:\n{}", get_stacktrace());
            if (s_previous_handler != nullptr) {
                return s_previous_handler(exp);
            } else {
                return EXCEPTION_EXECUTE_HANDLER;
            }
        });
    }();
#else   // _WIN32
    // Hook into signals.
    [[maybe_unused]] static bool registered = [] {
        register_for_signal<&s_sigsegv_handler>(SIGSEGV);
        register_for_signal<&s_sigbus_handler>(SIGBUS);
        register_for_signal<&s_sigill_handler>(SIGILL);
        return true;
    }();
#endif  // _WIN32
}

void install_uncaught_exception_handler() {
    // Technically always installed since we can't add it to existing threads, so instead
    // we just enable it.
    s_enable_uncaught_exception_handler.store(true, std::memory_order_relaxed);
}

}  // namespace dorado::utils
