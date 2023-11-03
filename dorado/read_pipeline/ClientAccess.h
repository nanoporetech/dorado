#pragma once

#include "Minimap2Options.h"

#include <string>

namespace dorado {

// TODO replace this dependency with type erasure (possibly by using an inversion of
// control container as we do in basecall_server)
struct AlignmentInfo {
    alignment::Minimap2Options minimap_options;
    std::string reference_file;
    int num_index_contruction_threads;
};

class ClientAccess {
public:
    virtual uint32_t client_id() const = 0;
    virtual AlignmentInfo alignment_info() const = 0;
};

class StandaloneClientAccess : public ClientAccess {
public:
    uint32_t client_id() const override { return -1; }
    AlignmentInfo alignment_info() const override { return {}; }
};

}  // namespace dorado