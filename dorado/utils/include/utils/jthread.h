#pragma once

#include <thread>

namespace dorado::utils {

/**
 * \brief Polyfill until std::jthread becomes available.
 */
class jthread : private std::thread {
public:
    using std::thread::join;
    using std::thread::thread;

    ~jthread() {
        if (joinable()) {
            join();
        }
    }
};

}  // namespace dorado::utils
