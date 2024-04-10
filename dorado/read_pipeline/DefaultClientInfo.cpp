#include "DefaultClientInfo.h"

#include "utils/types.h"

namespace dorado {

const AlignmentInfo DefaultClientInfo::empty_alignment_info{};
const BarcodingInfo DefaultClientInfo::empty_barcoding_info{};

DefaultClientInfo::DefaultClientInfo(const PolyTailSettings& polytail_settings)
        : m_poly_a_calculator(polytail_settings.active
                                      ? poly_tail::PolyTailCalculatorFactory::create(
                                                polytail_settings.is_rna,
                                                polytail_settings.config_file)
                                      : nullptr) {}

}  // namespace dorado
