#pragma once

#include <thread>
#include <utility>

namespace dorado::utils {

/**
 * \brief Polyfill until std::jthread becomes available.
 */
class jthread : private std::thread {
    jthread(const jthread &) = delete;
    jthread &operator=(const jthread &) = delete;

public:
    using std::thread::join;
    using std::thread::thread;

    jthread(jthread &&o) noexcept : jthread() { operator=(std::move(o)); }

    jthread &operator=(jthread &&o) noexcept {
        std::thread::operator=(std::move(o));
        return *this;
    }

    ~jthread() {
        if (joinable()) {
            join();
        }
    }
};

}  // namespace dorado::utils
