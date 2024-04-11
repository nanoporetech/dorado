#include "read_pipeline/ClientInfo.h"
#include "utils/types.h"

namespace dorado {

class FakeClientInfo : public dorado::ClientInfo {
    std::shared_ptr<const AdapterInfo> m_null_adapter_info{};
    dorado::AlignmentInfo m_align_info{};
    dorado::BarcodingInfo m_barcoding_info{};

public:
    void set_alignment_info(dorado::AlignmentInfo align_info) {
        m_align_info = std::move(align_info);
    }

    void set_barcoding_info(dorado::BarcodingInfo barcoding_info) {
        m_barcoding_info = std::move(barcoding_info);
    }

    int32_t client_id() const override { return 1; }

    const std::shared_ptr<const AdapterInfo>& adapter_info() const override {
        return m_null_adapter_info;
    }
    const dorado::AlignmentInfo& alignment_info() const override { return m_align_info; }
    const dorado::BarcodingInfo& barcoding_info() const override { return m_barcoding_info; }
    const poly_tail::PolyTailCalculator* poly_a_calculator() const override { return nullptr; }

    bool is_disconnected() const override { return false; }
};

}  // namespace dorado