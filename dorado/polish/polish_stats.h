#pragma once

#include "utils/stats.h"

#include <cstdint>
#include <mutex>
#include <string>

namespace dorado::polisher {

class PolishStats {
public:
    PolishStats() = default;

    void increment(const std::string& name) {
        std::unique_lock<std::mutex> lock(m_mtx);
        m_stats[name] += 1.0;
    }

    void add(const std::string& name, const double value) {
        std::unique_lock<std::mutex> lock(m_mtx);
        m_stats[name] += value;
    }

    void set(const std::string& name, const double value) {
        std::unique_lock<std::mutex> lock(m_mtx);
        m_stats[name] = value;
    }

    stats::NamedStats get_stats() const { return m_stats; }

private:
    stats::NamedStats m_stats;
    std::mutex m_mtx;
};

}  // namespace dorado::polisher
