#pragma once

#include "alignment/Minimap2Options.h"

#include <cstdint>
#include <string>

namespace dorado {

// TODO replace this explicit dependency on an alignment struct with type
// erasure (possibly by using an inversion ofcontrol container as we do in
// basecall_server)
struct AlignmentInfo {
    alignment::Minimap2Options minimap_options;
    std::string reference_file;
};

class ClientAccess {
public:
    virtual ~ClientAccess() = default;

    virtual const AlignmentInfo& alignment_info() const = 0;
    virtual int32_t client_id() const = 0;
    virtual bool is_disconnected() const = 0;
};

class StandaloneClientAccess final : public ClientAccess {
    inline static const AlignmentInfo empty_alignment_info{};

public:
    const AlignmentInfo& alignment_info() const override { return empty_alignment_info; }
    int32_t client_id() const override { return -1; }
    bool is_disconnected() const override { return false; }
};

}  // namespace dorado