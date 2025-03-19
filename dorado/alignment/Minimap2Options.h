#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace dorado::alignment {

constexpr inline std::string_view DEFAULT_MM_PRESET{"lr:hq"};

class Minimap2IdxOptHolder;
class Minimap2MapOptHolder;

struct Minimap2IndexOptions {
    Minimap2IndexOptions();
    std::shared_ptr<Minimap2IdxOptHolder> index_options;
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
    std::shared_ptr<Minimap2MapOptHolder> mapping_options;
};

bool operator<(const Minimap2MappingOptions& l, const Minimap2MappingOptions& r);
bool operator>(const Minimap2MappingOptions& l, const Minimap2MappingOptions& r);
bool operator<=(const Minimap2MappingOptions& l, const Minimap2MappingOptions& r);
bool operator>=(const Minimap2MappingOptions& l, const Minimap2MappingOptions& r);
bool operator==(const Minimap2MappingOptions& l, const Minimap2MappingOptions& r);
bool operator!=(const Minimap2MappingOptions& l, const Minimap2MappingOptions& r);

struct Minimap2Options : public Minimap2IndexOptions, public Minimap2MappingOptions {
    static std::optional<Minimap2Options> parse(const std::string& option_string,
                                                std::string& error_message);
};

bool operator==(const Minimap2Options& l, const Minimap2Options& r);
bool operator!=(const Minimap2Options& l, const Minimap2Options& r);

Minimap2Options create_options(const std::string& opt_str);
Minimap2Options create_dflt_options();  // the default preset is "lr:hq"
Minimap2Options create_preset_options(const std::string& preset);

}  // namespace dorado::alignment
