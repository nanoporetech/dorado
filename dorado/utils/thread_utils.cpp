#include "thread_utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
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

class SystemCPUUsage {
    struct Timings {
        std::size_t cpu, wall;
    };
    Timings m_last_timings{};

    std::optional<Timings> read_timings() {
        // Read the total CPU line from proc.
        std::ifstream proc_stat("/proc/stat");
        std::string first_line;
        if (!std::getline(proc_stat, first_line)) {
            spdlog::error("Failed to read /proc/stat");
            return std::nullopt;
        }
        std::istringstream cpu_line(first_line);

        // Read off everything until idle and store that separately.
        std::string cpu;
        std::size_t user = 0, nice = 0, system = 0, idle = 0;
        cpu_line >> cpu >> user >> nice >> system >> idle;
        if (cpu != "cpu" || !cpu_line) {
            spdlog::error("Failed to parse /proc/stat");
            return std::nullopt;
        }

        // Read off the rest of the timings.
        std::size_t cpu_total = user + nice + system;
        std::size_t timing = 0;
        while ((cpu_line >> timing)) {
            cpu_total += timing;
        }

        Timings timings;
        timings.cpu = cpu_total;
        timings.wall = cpu_total + idle;
        return timings;
    }

public:
    // Returns the average system CPU % since the last time this was called.
    std::optional<double> poll() {
        const auto current_timings = read_timings();
        if (!current_timings.has_value()) {
            return std::nullopt;
        }
        const double cpu_delta = static_cast<double>(current_timings->cpu - m_last_timings.cpu);
        const double wall_delta = static_cast<double>(current_timings->wall - m_last_timings.wall);
        m_last_timings = *current_timings;
        return cpu_delta / wall_delta;
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

static void init_load_balancers() {
#ifdef __linux__
    using Clock = std::chrono::steady_clock;
    // Kernel timings aren't updated that frequently, so we have to wait a while between polls.
    // Not doing so would cause the PID to overshoot since it'd be acting on stale data.
    static constexpr Clock::duration poll_every = std::chrono::milliseconds(500);

    static auto run_balancer_thread = [](std::size_t thread_id,
                                         const std::atomic<std::size_t>& threads_to_spin) {
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
            if (now - last_check_time > poll_every) {
                active = thread_id < threads_to_spin.load(std::memory_order_relaxed);
                last_check_time = now;
            }

            if (!active) {
                // Nothing to do.
                std::this_thread::sleep_for(poll_every);
                continue;
            }

            // Do some work, adjusting how long we work for.
            const auto work_duration = do_work(loop_count);
            if (work_duration < poll_every / 2) {
                loop_count *= 1.5;
            } else if (work_duration > poll_every) {
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
        double target_output = 0.7;  // ~70% gives good performance on our test machine.
        if (const char* envvar = getenv("DORADO_TARGET_USAGE"); envvar != nullptr) {
            target_output = std::atof(envvar);
        }
        target_output = std::clamp(target_output, 0.0, 1.0);
        spdlog::info("DORADO_TARGET_USAGE={}", target_output);

        // Monitor resource usage and keep us at the requested CPU usage.
        SystemCPUUsage cpu_usage;
        cpu_usage.poll();  // poll and discard initial input
        double previous_input = 0;
        while (true) {
            std::this_thread::sleep_for(poll_every);
            const auto current_output = cpu_usage.poll();

            // If we're not enabled then don't do any prediction.
            if (!s_spinners_enabled.load(std::memory_order_relaxed) ||
                !current_output.has_value()) {
                threads_to_spin.store(0, std::memory_order_relaxed);
                // Reset previous input so that we don't overload the machine
                // when we're next enabled.
                previous_input = 0;
                continue;
            }

            // Adjust the current resource usage.
            // TODO: PID controller
            double current_input = previous_input;
            current_input += (target_output - current_output.value()) / 2;
            current_input = std::clamp(current_input, 0.0, 1.0);

            const std::size_t num_cpus_to_spin = current_input * num_cpus;
            threads_to_spin.store(num_cpus_to_spin, std::memory_order_relaxed);
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
