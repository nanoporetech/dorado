#pragma once

#include "ClientInfo.h"
#include "poly_tail/poly_tail_calculator.h"

namespace dorado {

class DefaultClientInfo final : public ClientInfo {
    static const AlignmentInfo empty_alignment_info;
    const std::unique_ptr<const poly_tail::PolyTailCalculator> m_poly_a_calculator;

public:
    struct PolyTailSettings {
        bool active{false};
        bool is_rna{false};
        std::string config_file{};
    };

    DefaultClientInfo() = default;
    DefaultClientInfo(const PolyTailSettings& polytail_settings);
    ~DefaultClientInfo() = default;

    const AlignmentInfo& alignment_info() const override { return empty_alignment_info; }
    const poly_tail::PolyTailCalculator* poly_a_calculator() const override {
        return m_poly_a_calculator.get();
    };

    int32_t client_id() const override { return -1; }
    bool is_disconnected() const override { return false; }
};

}  // namespace dorado