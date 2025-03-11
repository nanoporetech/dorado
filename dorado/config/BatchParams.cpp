#include "BatchParams.h"

#include <spdlog/spdlog.h>
#include <toml.hpp>

#include <optional>
#include <stdexcept>

namespace dorado::config {

bool BatchParams::set_value(Value &self, const Value &other) {
    if (other.priority != Priority::FORCE && other.priority <= self.priority) {
        return false;
    }
    if (self.val == other.val) {
        return false;
    }
    if (other.val < 0) {
        throw std::runtime_error("BatchParams::set_value value must be positive integer");
    }

    self.val = other.val;
    self.priority = other.priority;
    return true;
};

void BatchParams::update(const std::filesystem::path &path) {
    const auto config_toml = toml::parse(path / "config.toml");
    if (!config_toml.contains("basecaller")) {
        return;
    }

    const auto b = toml::find(config_toml, "basecaller");

    // Get value from config asserting that it's valid
    auto parse = [&b](const std::string &name) -> std::optional<int> {
        if (!b.contains(name)) {
            return std::nullopt;
        }
        const int value = toml::find<int>(b, name);
        if (value < 0) {
            spdlog::warn(
                    "invalid config - basecaller.{} must not be negative - got: {} - using default",
                    name, value);
            return std::nullopt;
        }
        return std::optional(value);
    };

    if (b.contains("batchsize")) {
        spdlog::trace(
                "config parameter basecaller.batchsize is ignored "
                "- to set it, use the '--batchsize' command line argument");
    }

    update(Priority::CONFIG, parse("chunksize"), parse("overlap"), std::nullopt);
};

void BatchParams::update(Priority priority,
                         std::optional<int> chunksize,
                         std::optional<int> overlap,
                         std::optional<int> batchsize) {
    auto upd = [&](Value &self, const std::optional<int> val, const std::string &name) {
        if (!val.has_value()) {
            return;
        }
        const int before_val = self.val;
        if (set_value(self, Value{val.value(), priority})) {
            spdlog::trace("Parsed: {} {} -> {}", name, before_val, val.value());
        }
    };

    upd(m_chunk_size, chunksize, "chunksize");
    upd(m_overlap, overlap, "overlap");
    upd(m_batch_size, batchsize, "batchsize");
}

void BatchParams::update(const BatchParams &other) {
    // Apply updated value with other.priority
    auto merge = [&](Value &self, const Value &oth, const std::string &name) {
        if (set_value(self, oth)) {
            spdlog::trace("Merged: {} {} -> {}", name, self.val, oth.val);
        }
    };

    merge(m_chunk_size, other.m_chunk_size, "chunksize");
    merge(m_overlap, other.m_overlap, "overlap");
    merge(m_batch_size, other.m_batch_size, "batchsize");
}

void BatchParams::normalise(int chunk_size_granularity, int stride) {
    // Make sure overlap is a multiple of `stride`, and greater than 0
    const int old_overlap = m_overlap.val;
    const int new_overlap = std::max(1, old_overlap / stride) * stride;
    if (set_value(m_overlap, Value{new_overlap, Priority::FORCE})) {
        spdlog::info("Normalised: overlap {} -> {}", old_overlap, new_overlap);
    }

    // Make sure chunk size is a multiple of `chunk_size_granularity`, and greater than `overlap`
    const int old_chunk_size = m_chunk_size.val;
    const int min_chunk_size = new_overlap + chunk_size_granularity - 1;
    const int new_chunk_size = (std::max(min_chunk_size, old_chunk_size) / chunk_size_granularity) *
                               chunk_size_granularity;
    if (set_value(m_chunk_size, Value{new_chunk_size, Priority::FORCE})) {
        spdlog::info("Normalised: chunksize {} -> {}", old_chunk_size, new_chunk_size);
    }
}

std::string BatchParams::to_string() const {
    std::ostringstream oss;
    // clang-format off
    oss << "BatchParams {"
        << " chunk_size:" << m_chunk_size.val 
        << " overlap:" << m_overlap.val
        << " batch_size:" << m_batch_size.val << "}";
    return oss.str();
    // clang-format on
}

}  // namespace dorado::config