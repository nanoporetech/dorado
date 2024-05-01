#pragma once

#include "ClientInfo.h"
#include "utils/types.h"

#include <memory>

namespace dorado {

class DefaultClientInfo final : public ClientInfo {
    ContextContainer m_contexts{};

public:
    DefaultClientInfo() = default;
    ~DefaultClientInfo() = default;

    // Implementation of ClientInfo interface
    int32_t client_id() const override { return -1; }
    bool is_disconnected() const override { return false; }
    ContextContainer& contexts() override { return m_contexts; }
    const ContextContainer& contexts() const override { return m_contexts; }
};

}  // namespace dorado