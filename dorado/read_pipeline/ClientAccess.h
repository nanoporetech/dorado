#pragma once

#include "alignment/Minimap2Options.h"

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
    virtual uint32_t client_id() const = 0;
    virtual const AlignmentInfo& alignment_info() const = 0;
};

class StandaloneClientAccess : public ClientAccess {
public:
    uint32_t client_id() const override { return -1; }
    const AlignmentInfo& alignment_info() const override { return {}; }
};

}  // namespace dorado