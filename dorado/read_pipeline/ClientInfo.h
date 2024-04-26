#pragma once

#include "alignment/Minimap2Options.h"
#include "context_container.h"

#include <cstdint>
#include <memory>
#include <string>

namespace dorado {

struct AdapterInfo;

class ClientInfo {
public:
    virtual ~ClientInfo() = default;

    // Change to a reference when we remove the default from AdapterDetectorNode
    // until then need to know if set or not, in order to know whether to override
    // the node's default.
    virtual const std::shared_ptr<AdapterInfo>& adapter_info() const = 0;

    virtual int32_t client_id() const = 0;
    virtual bool is_disconnected() const = 0;

    virtual ContextContainer& contexts() = 0;
    virtual const ContextContainer& contexts() const = 0;
};

}  // namespace dorado