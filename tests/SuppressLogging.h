#pragma once

#include <spdlog/spdlog.h>

#include <memory>

namespace dorado::tests {

/// <summary>
/// RAII helper to disable the default logger for the current scope
/// </summary>
class [[nodiscard]] SuppressLogging final {
    const std::shared_ptr<spdlog::logger> m_logger;
    const spdlog::level::level_enum m_level;

public:
    SuppressLogging() : m_logger(spdlog::default_logger()), m_level(m_logger->level()) {
        m_logger->set_level(spdlog::level::off);
    }
    ~SuppressLogging() { m_logger->set_level(m_level); }

    SuppressLogging(const SuppressLogging &) = delete;
    SuppressLogging &operator=(const SuppressLogging &) = delete;
    SuppressLogging(SuppressLogging &&) = delete;
    SuppressLogging &operator=(SuppressLogging &&) = delete;
};

}  // namespace dorado::tests
