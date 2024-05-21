#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>

namespace dorado::alignment {

constexpr inline std::string_view DEFAULT_MM_PRESET{"lr:hq"};

namespace minimap2 {
class IdxOptHolder;
class MapOptHolder;
}  // namespace minimap2

struct Minimap2IndexOptions {
    Minimap2IndexOptions();
    std::shared_ptr<minimap2::IdxOptHolder> index_options;
    std::string junc_bed;
};

bool operator<(const Minimap2IndexOptions& l, const Minimap2IndexOptions& r);
bool operator>(const Minimap2IndexOptions& l, const Minimap2IndexOptions& r);
bool operator<=(const Minimap2IndexOptions& l, const Minimap2IndexOptions& r);
bool operator>=(const Minimap2IndexOptions& l, const Minimap2IndexOptions& r);
bool operator==(const Minimap2IndexOptions& l, const Minimap2IndexOptions& r);
bool operator!=(const Minimap2IndexOptions& l, const Minimap2IndexOptions& r);

struct Minimap2MappingOptions {
    Minimap2MappingOptions();
    std::shared_ptr<minimap2::MapOptHolder> mapping_options;
};

bool operator<(const Minimap2MappingOptions& l, const Minimap2MappingOptions& r);
bool operator>(const Minimap2MappingOptions& l, const Minimap2MappingOptions& r);
bool operator<=(const Minimap2MappingOptions& l, const Minimap2MappingOptions& r);
bool operator>=(const Minimap2MappingOptions& l, const Minimap2MappingOptions& r);
bool operator==(const Minimap2MappingOptions& l, const Minimap2MappingOptions& r);
bool operator!=(const Minimap2MappingOptions& l, const Minimap2MappingOptions& r);

struct Minimap2Options : public Minimap2IndexOptions, public Minimap2MappingOptions {};

bool operator==(const Minimap2Options& l, const Minimap2Options& r);
bool operator!=(const Minimap2Options& l, const Minimap2Options& r);

Minimap2Options create_dflt_options();  // the default preset is "lr:hq"
Minimap2Options create_preset_options(const std::string& preset);

}  // namespace dorado::alignment
