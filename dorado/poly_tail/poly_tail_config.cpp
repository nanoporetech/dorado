#include "poly_tail_config.h"

#include "utils/sequence_utils.h"

#include <toml.hpp>
#include <toml/value.hpp>

namespace dorado::poly_tail {

PolyTailConfig prepare_config(const std::string* config_file) {
    PolyTailConfig config;

    if (config_file != nullptr) {
        const toml::value config_toml = toml::parse(*config_file);

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
                config.plasmid_front_flank =
                        toml::find<std::string>(anchors, "plasmid_front_flank");
                config.plasmid_rear_flank = toml::find<std::string>(anchors, "plasmid_rear_flank");
                config.is_plasmid = true;
            }
        }

        if (config_toml.contains("tail")) {
            const auto& tail = toml::find(config_toml, "tail");

            if (tail.contains("tail_interrupt_length")) {
                config.tail_interrupt_length = toml::find<int>(tail, "tail_interrupt_length");
            }
        }
    }

    if (!config.front_primer.empty()) {
        config.rc_front_primer = dorado::utils::reverse_complement(config.front_primer);
    }
    if (!config.rear_primer.empty()) {
        config.rc_rear_primer = dorado::utils::reverse_complement(config.rear_primer);
    }
    if (!config.plasmid_front_flank.empty()) {
        config.rc_plasmid_front_flank =
                dorado::utils::reverse_complement(config.plasmid_front_flank);
    }
    if (!config.plasmid_rear_flank.empty()) {
        config.rc_plasmid_rear_flank = dorado::utils::reverse_complement(config.plasmid_rear_flank);
    }

    return config;
}

}  // namespace dorado::poly_tail
