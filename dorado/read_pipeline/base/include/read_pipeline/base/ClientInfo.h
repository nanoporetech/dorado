#pragma once

#include <cstdint>

namespace dorado {

class ContextContainer;

class ClientInfo {
public:
    virtual ~ClientInfo() = default;

    virtual int32_t client_id() const = 0;
    virtual bool is_disconnected() const = 0;

    virtual ContextContainer& contexts() = 0;
    virtual const ContextContainer& contexts() const = 0;
};

}  // namespace dorado