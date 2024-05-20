#include "Minimap2Options.h"

#include "minimap2_wrappers.h"

namespace dorado::alignment {

Minimap2IndexOptions::Minimap2IndexOptions() {
    index_options = std::make_shared<minimap2::IdxOptHolder>();
}

Minimap2MappingOptions::Minimap2MappingOptions() {
    mapping_options = std::make_shared<minimap2::MapOptHolder>();
}

}  // namespace dorado::alignment