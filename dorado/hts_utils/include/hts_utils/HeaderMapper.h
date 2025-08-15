#pragma once

#include <htslib/sam.h>

#include <filesystem>
#include <vector>

namespace dorado::utils {

class HeaderMapper {
public:
    /** Construct a mapping of structured output key to merged headers 
     *  @param inputs Collection of input file hts file inputs to map
     *  @param strip_alignment If set, no SQ lines will be included in the
     *         merged header, and no checks will be made for SQ conflicts. 
     */
    HeaderMapper(const std::vector<std::filesystem::path>& inputs, bool strip_alignment);

private:
    const bool m_strip_alignment;

    void process(const std::vector<std::filesystem::path>& inputs);
};

}  // namespace dorado::utils
