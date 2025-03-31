#include "poly_tail_config.h"

#include "utils/sequence_utils.h"

#include <toml.hpp>

#include <filesystem>
#include <fstream>
#include <istream>
#include <sstream>
#include <string>
#include <unordered_set>

namespace dorado::poly_tail {
namespace {

PolyTailConfig update_config(const toml::value& config_toml, PolyTailConfig config) {
    if (config_toml.contains("barcode_id")) {
        config.barcode_id = toml::find<std::string>(config_toml, "barcode_id");
    }

    if (config_toml.contains("anchors")) {
        const auto& anchors = toml::find(config_toml, "anchors");

        if (anchors.contains("front_primer") || anchors.contains("rear_primer")) {
            if (!(anchors.contains("front_primer") && anchors.contains("rear_primer"))) {
                throw std::runtime_error(
                        "Both front_primer and rear_primer must be provided in the PolyA "
                        "configuration file.");
            }
            config.front_primer = toml::find<std::string>(anchors, "front_primer");
            config.rear_primer = toml::find<std::string>(anchors, "rear_primer");
        }

        if (anchors.contains("plasmid_front_flank") || anchors.contains("plasmid_rear_flank")) {
            if (!(anchors.contains("plasmid_front_flank") &&
                  anchors.contains("plasmid_rear_flank"))) {
                throw std::runtime_error(
                        "Both plasmid_front_flank and plasmid_rear_flank must be provided in "
                        "the PolyA configuration file.");
            }
            config.plasmid_front_flank = toml::find<std::string>(anchors, "plasmid_front_flank");
            config.plasmid_rear_flank = toml::find<std::string>(anchors, "plasmid_rear_flank");
            config.is_plasmid = true;
            config.flank_threshold = 0.85f;  // stricter default for plasmids
        }

        if (anchors.contains("primer_window")) {
            config.primer_window = toml::find<int>(anchors, "primer_window");
            if (config.primer_window <= 0) {
                throw std::runtime_error("primer_window size needs to be > 0, given " +
                                         std::to_string(config.primer_window));
            }
        }

        if (anchors.contains("min_primer_separation")) {
            config.min_primer_separation = toml::find<int>(anchors, "min_primer_separation");
            if (config.min_primer_separation <= 0) {
                throw std::runtime_error("min_primer_separation size needs to be > 0, given " +
                                         std::to_string(config.min_primer_separation));
            }
        }
    }

    if (config_toml.contains("threshold")) {
        const auto& threshold = toml::find(config_toml, "threshold");
        if (threshold.contains("flank_threshold")) {
            config.flank_threshold = toml::find<float>(threshold, "flank_threshold");
        }
    }

    if (config_toml.contains("tail")) {
        const auto& tail = toml::find(config_toml, "tail");

        if (tail.contains("tail_interrupt_length")) {
            config.tail_interrupt_length = toml::find<int>(tail, "tail_interrupt_length");
        }
    }

    config.rc_front_primer = utils::reverse_complement(config.front_primer);
    config.rc_rear_primer = utils::reverse_complement(config.rear_primer);
    config.rc_plasmid_front_flank = utils::reverse_complement(config.plasmid_front_flank);
    config.rc_plasmid_rear_flank = utils::reverse_complement(config.plasmid_rear_flank);

    return config;
}

void add_configs(const toml::value& config_toml, std::vector<PolyTailConfig>& configs) {
    // add the default config
    auto default_config = update_config(config_toml, PolyTailConfig{});
    if (!default_config.barcode_id.empty()) {
        throw std::runtime_error("Default poly tail config must not specify barcode_id.");
    }

    // get override configs
    if (config_toml.contains("overrides")) {
        const std::vector<toml::value> overrides = toml::find(config_toml, "overrides").as_array();
        std::unordered_set<std::string> ids;
        for (auto& override_toml : overrides) {
            auto override = update_config(override_toml, default_config);
            ids.insert(override.barcode_id);
            configs.push_back(std::move(override));
        }
        if (ids.count("") != 0) {
            throw std::runtime_error("Missing barcode_id in override poly tail configuration.");
        }
        if (ids.size() != overrides.size()) {
            throw std::runtime_error("Duplicate barcode_id found in poly tail config file.");
        }
    }

    configs.push_back(std::move(default_config));
}

}  // namespace

std::vector<PolyTailConfig> prepare_configs(std::istream& is) {
    const toml::value config_toml = toml::parse(is);
    std::vector<PolyTailConfig> configs;
    add_configs(config_toml, configs);
    return configs;
}

std::vector<PolyTailConfig> prepare_configs(const std::string& config_file) {
    if (!config_file.empty()) {
        if (!std::filesystem::exists(config_file) ||
            !std::filesystem::is_regular_file(config_file)) {
            throw std::runtime_error("PolyA config file doesn't exist at " + config_file);
        }
        std::ifstream file(config_file);  // Open the file for reading
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file " + config_file);
        }

        // Read the file contents into a string
        std::stringstream buffer;
        buffer << file.rdbuf();
        return prepare_configs(buffer);
    } else {
        std::stringstream buffer("");
        return prepare_configs(buffer);
    }
}

}  // namespace dorado::poly_tail
