#include "thread_utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

#ifdef _WIN32
#include <Windows.h>
#elif defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace dorado::utils {

namespace {

class CPUUsage {
    timespec last_cpu_time{};
    timespec last_wall_time{};

public:
    // Returns the average CPU usage since the last time this was called.
    double poll() {
        timespec current_cpu_time{};
        timespec current_wall_time{};
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &current_cpu_time);
        clock_gettime(CLOCK_MONOTONIC, &current_wall_time);

        const double cpu_time = (current_cpu_time.tv_sec - last_cpu_time.tv_sec) +
                                1e-9 * (current_cpu_time.tv_nsec - last_cpu_time.tv_nsec);
        const double wall_time = (current_wall_time.tv_sec - last_wall_time.tv_sec) +
                                 1e-9 * (current_wall_time.tv_nsec - last_wall_time.tv_nsec);

        last_cpu_time = current_cpu_time;
        last_wall_time = current_wall_time;

        return cpu_time / wall_time;
    }
};

std::atomic<bool> s_spinners_enabled{false};

}  // namespace

void set_thread_name(const char* name) {
#if defined(_WIN32)
    // There is an alternative thread name mechanism based on throwing an exception. We don't bother
    // with it because it only works when the process is being run under the Visual Studio debugger
    // at the time the thread name is set, and the method below works with the Windows/VS versions
    // developers are likely to have anyway.

    // See: https://randomascii.wordpress.com/2015/10/26/thread-naming-in-windows-time-for-something-better/
    typedef HRESULT(WINAPI * SetThreadDescription)(HANDLE hThread, PCWSTR lpThreadDescription);

    // The SetThreadDescription API works even if no debugger is attached. It requires Windows 10
    // build 1607 or later. The thread name set this way will only be picked up by Visual Studio
    // 2017 version 15.6 or later. See
    // https://docs.microsoft.com/en-gb/visualstudio/debugger/how-to-set-a-thread-name-in-native-code
    auto set_thread_description_func = reinterpret_cast<SetThreadDescription>(
            ::GetProcAddress(::GetModuleHandleA("Kernel32.dll"), "SetThreadDescription"));

    if (set_thread_description_func) {
        std::array<wchar_t, 64> wide_name;
        std::size_t output_size = 0;
        mbstowcs_s(&output_size, wide_name.data(), wide_name.size(), name, _TRUNCATE);
        set_thread_description_func(::GetCurrentThread(), wide_name.data());
    }

#else

    // Name is limited to 16 chars including null terminator.
    std::array<char, 16> limited_name;
    const auto name_len = std::min(std::size_t(15), strlen(name));
    std::copy(name, name + name_len, limited_name.begin());
    limited_name[name_len] = '\0';

#if defined(__APPLE__)
    pthread_setname_np(limited_name.data());
#else
    pthread_setname_np(pthread_self(), limited_name.data());
#endif
#endif
}

#ifdef __linux__
static void pin_current_thread_to_cpu(std::size_t cpu_id) {
    const auto num_threads = std::thread::hardware_concurrency();
    cpu_set_t* cpuset = CPU_ALLOC(num_threads);
    const auto cpusetsize = CPU_ALLOC_SIZE(num_threads);
    CPU_ZERO(cpuset);
    CPU_SET(cpu_id, cpuset);

    const auto handle = pthread_self();
    const int err = pthread_setaffinity_np(handle, cpusetsize, cpuset);
    if (err != 0) {
        spdlog::error("[{}] Failed to pin thread to core {}: {}", handle, cpu_id, err);
    }

    CPU_FREE(cpuset);
}
#endif

static void init_load_balancers() {
#ifdef __linux__
    using Clock = std::chrono::steady_clock;
    static constexpr Clock::duration check_every = std::chrono::milliseconds(500);

    static const bool pin_balancers = [] {
        const char* envvar = getenv("pin_balancers");
        const bool should = envvar != nullptr && envvar[0] == '1';
        spdlog::info("pin_balancers={} ({})", envvar ? envvar : "", should ? "yes" : "no");
        return should;
    }();

    static auto run_balancer_thread = [](std::size_t thread_id,
                                         const std::atomic<std::size_t>& threads_to_spin) {
        if (pin_balancers) {
            // TODO: should spread them out, something like this:
            //   thread: 0 1 2 3 4 5 6 7
            //   cpu:    0 4 2 6 1 5 3 7
            // Looks like https://oeis.org/A030109
            pin_current_thread_to_cpu(thread_id);
        }

        // Helper to do some work.
        auto do_work = [](std::size_t loop_count) {
            auto start = Clock::now();
            volatile float val = 1 + rand();
            for (std::size_t i = 0; i < loop_count; i++) {
                val = std::sqrt(val) + 1.f;
            }
            return Clock::now() - start;
        };

        constexpr std::size_t min_loop_count = 1'000'000;
        std::size_t loop_count = min_loop_count;
        auto last_check_time = Clock::now();
        bool active = false;
        while (true) {
            // See if we should be active.
            const auto now = Clock::now();
            if (now - last_check_time > check_every) {
                active = thread_id < threads_to_spin.load(std::memory_order_relaxed);
                last_check_time = now;
            }

            if (!active) {
                // Nothing to do.
                std::this_thread::sleep_for(check_every);
                continue;
            }

            // Do some work, adjusting how long we work for.
            const auto work_duration = do_work(loop_count);
            if (work_duration < check_every / 2) {
                loop_count *= 1.5;
            } else if (work_duration > check_every) {
                loop_count /= 2;
            }
            loop_count = std::max(min_loop_count, loop_count);
        }
    };

    static auto run_monitor_thread = [] {
        const auto num_cpus = std::thread::hardware_concurrency();
        std::atomic<std::size_t> threads_to_spin{0};

        // Kick off balancer threads.
        std::vector<std::thread> balancers(num_cpus);
        for (std::size_t thread_id = 0; thread_id < num_cpus; thread_id++) {
            balancers[thread_id] = std::thread([thread_id, &threads_to_spin] {
                run_balancer_thread(thread_id, threads_to_spin);
            });
        }

        // Read off target CPU usage.
        double target_output = 0.7;
        if (const char* envvar = getenv("target_output"); envvar != nullptr) {
            target_output = std::atof(envvar);
        }
        target_output = std::clamp(target_output, 0.0, 1.0);
        target_output *= num_cpus;
        spdlog::info("target_output={}", target_output);

        // Monitor resource usage and keep us at the requested CPU usage.
        CPUUsage cpu_usage;
        cpu_usage.poll();  // poll and discard initial input
        double previous_input = 0;
        while (true) {
            std::this_thread::sleep_for(check_every);

            // If we're not enabled then don't do any prediction.
            if (!s_spinners_enabled.load(std::memory_order_relaxed)) {
                threads_to_spin.store(0, std::memory_order_relaxed);
                // Poll but discard.
                cpu_usage.poll();
                previous_input = 0;
                continue;
            }

            // Adjust the current resource usage.
            // TODO: PID controller
            const double current_output = cpu_usage.poll();
            double current_input = previous_input;
            current_input += (target_output - current_output) / 2;
            current_input = std::clamp(current_input, 0.0, static_cast<double>(num_cpus));

            threads_to_spin.store(current_input, std::memory_order_relaxed);
            previous_input = current_input;
        }
    };

    [[maybe_unused]] static auto thread_starter = [] {
        std::thread(run_monitor_thread).detach();
        return true;
    }();
#endif
}

void start_busy_work() {
    // Init them if they haven't been already.
    init_load_balancers();
    s_spinners_enabled.store(true, std::memory_order_relaxed);
}

void stop_busy_work() { s_spinners_enabled.store(false, std::memory_order_relaxed); }

}  // namespace dorado::utils
