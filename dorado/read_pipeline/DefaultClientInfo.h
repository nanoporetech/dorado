#pragma once

#include "ClientInfo.h"
#include "poly_tail/poly_tail_calculator.h"
#include "utils/types.h"

#include <memory>

namespace dorado {

class DefaultClientInfo : public ClientInfo {
    std::shared_ptr<AlignmentInfo> m_alignment_info = std::make_shared<AlignmentInfo>();
    std::shared_ptr<BarcodingInfo> m_barcoding_info = std::make_shared<BarcodingInfo>();
    const std::unique_ptr<const poly_tail::PolyTailCalculator> m_poly_a_calculator;
    std::shared_ptr<AdapterInfo> m_adapter_info{};

public:
    struct PolyTailSettings {
        bool active{false};
        bool is_rna{false};
        std::string config_file{};
    };

    DefaultClientInfo() = default;
    DefaultClientInfo(const PolyTailSettings& polytail_settings);
    ~DefaultClientInfo() = default;

    const std::shared_ptr<AdapterInfo>& adapter_info() const override { return m_adapter_info; }
    const AlignmentInfo& alignment_info() const override { return *m_alignment_info; }
    const BarcodingInfo& barcoding_info() const override { return *m_barcoding_info; }
    const poly_tail::PolyTailCalculator* poly_a_calculator() const override {
        return m_poly_a_calculator.get();
    };
    int32_t client_id() const override { return -1; }
    bool is_disconnected() const override { return false; }

    void set_alignment_info(std::shared_ptr<AlignmentInfo>& alignment_info) {
        m_alignment_info = alignment_info;
    }
    void set_barcoding_info(std::shared_ptr<BarcodingInfo>& barcoding_info) {
        m_barcoding_info = barcoding_info;
    }
    void set_adapter_info(std::shared_ptr<AdapterInfo>& adapter_info) {
        m_adapter_info = adapter_info;
    }
};

}  // namespace dorado