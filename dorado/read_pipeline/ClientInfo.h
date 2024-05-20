#pragma once

#include "alignment/Minimap2Options.h"

#include <cstdint>
#include <string>

namespace dorado {

// TODO replace this explicit dependency on an alignment struct with type
// erasure (possibly by using an inversion of control container as we do in
// basecall_server)
struct AlignmentInfo {
    alignment::Minimap2Options minimap_options;
    std::string reference_file;
    std::string alignment_header;
};

class ClientInfo {
public:
    virtual ~ClientInfo() = default;

    virtual const AlignmentInfo& alignment_info() const = 0;
    virtual int32_t client_id() const = 0;
    virtual bool is_disconnected() const = 0;
};

}  // namespace dorado