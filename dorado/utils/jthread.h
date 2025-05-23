#pragma once

#include <memory>
#include <thread>

namespace dorado::utils {

/**
 * \brief Workaround until std::jthread becomes available.
 *          Usage:
 *              std::shared_ptr<std::thread> thread_sample_producer = make_jthread(std::thread(
 *                  &worker, std::cref(param1), std::cref(param2));
 */
inline std::shared_ptr<std::thread> make_jthread(std::thread&& t) {
    return std::shared_ptr<std::thread>(new std::thread(std::move(t)), [](std::thread* tp) {
        if (tp->joinable()) {
            tp->join();
        }
        delete tp;
    });
}

}  // namespace dorado::utils