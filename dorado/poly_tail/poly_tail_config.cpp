#include "poly_tail_config.h"

#include "utils/sequence_utils.h"

#include <toml.hpp>
#include <toml/value.hpp>

#include <fstream>
#include <istream>
#include <sstream>
#include <string>

namespace dorado::poly_tail {

PolyTailConfig prepare_config(std::istream& is) {
    PolyTailConfig config;

    const toml::value config_toml = toml::parse(is);

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
            config.flank_threshold = 0.15;  // reduced default for plasmids
        }

        if (anchors.contains("primer_window")) {
            config.primer_window = toml::find<int>(anchors, "primer_window");
            if (config.primer_window <= 0) {
                throw std::runtime_error("primer_window size needs to be > 0, given " +
                                         std::to_string(config.primer_window));
            }
        }
    }

    if (config_toml.contains("threshold")) {
        const auto& threshold = toml::find(config_toml, "threshold");
        if (threshold.contains("flank_threshold ")) {
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

PolyTailConfig prepare_config(const std::string& config_file) {
    if (!config_file.empty()) {
        std::ifstream file(config_file);  // Open the file for reading
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file " + config_file);
        }

        // Read the file contents into a string
        std::stringstream buffer;
        buffer << file.rdbuf();
        return prepare_config(buffer);
    } else {
        std::stringstream buffer("");
        return prepare_config(buffer);
    }
}

}  // namespace dorado::poly_tail
