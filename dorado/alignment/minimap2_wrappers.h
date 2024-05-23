#pragma once

#include <minimap.h>

// Helper structs allowing the types to be forward declared
// minimap uses typedefs to anonymous structs which prevents
// forward declarations.
namespace dorado::alignment {

class Minimap2IdxOptHolder {
    mm_idxopt_t m_index_options;

public:
    mm_idxopt_t& get() { return m_index_options; }
    const mm_idxopt_t& get() const { return m_index_options; }
};

class Minimap2MapOptHolder {
    mm_mapopt_t m_mapping_options;

public:
    mm_mapopt_t& get() { return m_mapping_options; }
    const mm_mapopt_t& get() const { return m_mapping_options; }
};

}  // namespace dorado::alignment