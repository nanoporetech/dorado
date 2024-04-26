#include "DefaultClientInfo.h"

namespace dorado {

const std::shared_ptr<AdapterInfo>& DefaultClientInfo::adapter_info() const {
    return m_adapter_info;
}

int32_t DefaultClientInfo::client_id() const { return -1; }

bool DefaultClientInfo::is_disconnected() const { return false; }

void DefaultClientInfo::set_adapter_info(std::shared_ptr<AdapterInfo> adapter_info) {
    m_adapter_info = std::move(adapter_info);
}

ContextContainer& DefaultClientInfo::contexts() { return m_contexts; }

const ContextContainer& DefaultClientInfo::contexts() const { return m_contexts; }

}  // namespace dorado
