#pragma once

#include "ClientInfo.h"
#include "utils/types.h"

#include <memory>

namespace dorado {

class DefaultClientInfo : public ClientInfo {
    std::shared_ptr<AlignmentInfo> m_alignment_info = std::make_shared<AlignmentInfo>();
    std::shared_ptr<BarcodingInfo> m_barcoding_info = std::make_shared<BarcodingInfo>();
    std::shared_ptr<AdapterInfo> m_adapter_info{};
    ContextContainer m_contexts{};

public:
    DefaultClientInfo() = default;
    ~DefaultClientInfo() = default;

    void set_alignment_info(std::shared_ptr<AlignmentInfo> alignment_info);
    void set_adapter_info(std::shared_ptr<AdapterInfo> adapter_info);

    // Implementation of ClientInfo interface
    const std::shared_ptr<AdapterInfo>& adapter_info() const override;
    const AlignmentInfo& alignment_info() const override;
    int32_t client_id() const override;
    bool is_disconnected() const override;
    ContextContainer& contexts() override;
    const ContextContainer& contexts() const override;
};

}  // namespace dorado