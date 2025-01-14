#include "poly_tail_calculator_selector.h"

#include "poly_tail_calculator.h"
#include "poly_tail_config.h"
#include "utils/types.h"

#include <cassert>
#include <fstream>
#include <sstream>

namespace dorado::poly_tail {

PolyTailCalculatorSelector::PolyTailCalculatorSelector(const std::filesystem::path& config,
                                                       bool is_rna,
                                                       bool is_rna_adapter,
                                                       float speed_calibration,
                                                       float offset_calibration) {
    if (config.empty()) {
        std::stringstream buffer("");
        init(buffer, is_rna, is_rna_adapter, speed_calibration, offset_calibration);
        return;
    }

    if (!std::filesystem::exists(config) || !std::filesystem::is_regular_file(config)) {
        throw std::runtime_error("PolyA config file doesn't exist at " + config.string());
    }

    std::ifstream file(config);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file " + config.string());
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    init(buffer, is_rna, is_rna_adapter, speed_calibration, offset_calibration);
}

PolyTailCalculatorSelector::PolyTailCalculatorSelector(std::istream& config_stream,
                                                       bool is_rna,
                                                       bool is_rna_adapter,
                                                       float speed_calibration,
                                                       float offset_calibration) {
    init(config_stream, is_rna, is_rna_adapter, speed_calibration, offset_calibration);
}

void PolyTailCalculatorSelector::init(std::istream& config_stream,
                                      bool is_rna,
                                      bool is_rna_adapter,
                                      float speed_calibration,
                                      float offset_calibration) {
    assert(speed_calibration > 0.f);
    auto configs = prepare_configs(config_stream);
    m_default = PolyTailCalculatorFactory::create(configs.back(), is_rna, is_rna_adapter,
                                                  speed_calibration, offset_calibration);
    configs.pop_back();

    std::lock_guard<std::mutex> lock(m_lut_mutex);
    for (const auto& config : configs) {
        m_lut[config.barcode_id] = PolyTailCalculatorFactory::create(
                config, is_rna, is_rna_adapter, speed_calibration, offset_calibration);
    }
}

// Return the barcode-specific configuration if one has been provided, otherwise the default.
// If any barcode-specific configurations are present, do not attempt to estimate
// for unclassified reads - better to give no result than a wrong result in this case.
std::shared_ptr<const PolyTailCalculator> PolyTailCalculatorSelector::get_calculator(
        const std::string& name) const {
    std::lock_guard<std::mutex> lock(m_lut_mutex);
    auto it = m_lut.find(name);
    return (it == std::end(m_lut)) ? (name == UNCLASSIFIED && !m_lut.empty() ? nullptr : m_default)
                                   : it->second;
}

}  // namespace dorado::poly_tail
