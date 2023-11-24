#pragma once

#include "ClientInfo.h"

namespace dorado {

class DefaultClientInfo final : public ClientInfo {
    inline static const AlignmentInfo empty_alignment_info{};

public:
    const AlignmentInfo& alignment_info() const override { return empty_alignment_info; }
    int32_t client_id() const override { return -1; }
    bool is_disconnected() const override { return false; }
};

}  // namespace dorado