#include "DefaultClientInfo.h"

#include "poly_tail/poly_tail_calculator.h"

namespace dorado {

const AlignmentInfo DefaultClientInfo::empty_alignment_info{};

DefaultClientInfo::DefaultClientInfo(const PolyTailSettings& polytail_settings)
        : m_poly_a_calculator(polytail_settings.active
                                      ? poly_tail::PolyTailCalculatorFactory::create(
                                                polytail_settings.is_rna,
                                                polytail_settings.config_file)
                                      : nullptr) {}

// out-of-line definition due to unique_ptr to forward declared class
DefaultClientInfo::~DefaultClientInfo() = default;

}  // namespace dorado
