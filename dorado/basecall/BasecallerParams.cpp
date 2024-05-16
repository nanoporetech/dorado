#include "BasecallerParams.h"

#include "utils/parameters.h"

#include <spdlog/spdlog.h>
#include <toml.hpp>

#include <optional>
#include <stdexcept>

namespace dorado::basecall {

bool BasecallerParams::set_value(Value &self, const Value &other) {
    if (other.priority != Priority::FORCE && other.priority <= self.priority) {
        return false;
    }
    if (self.val == other.val) {
        return false;
    }
    if (other.val < 0) {
        throw std::runtime_error("BasecallerParams::set_value value must be positive integer");
    }

    self.val = other.val;
    self.priority = other.priority;
    return true;
};

void BasecallerParams::update(const std::filesystem::path &path) {
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

void BasecallerParams::update(Priority priority,
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

void BasecallerParams::update(const BasecallerParams &other) {
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

void BasecallerParams::normalise(size_t divisor) {
    const int div = static_cast<int>(divisor);

    // Apply normalised value with FORCE
    auto normalise_param = [&, div](Value &self, const std::string &name) {
        const int before_val = self.val;
        const int new_val = (self.val / div) * div;
        if (set_value(self, Value{new_val, Priority::FORCE})) {
            spdlog::info("Normalised: {} {} -> {}", name, before_val, new_val);
        }
    };

    normalise_param(m_chunk_size, "chunksize");
    normalise_param(m_overlap, "overlap");
}

}  // namespace dorado::basecall