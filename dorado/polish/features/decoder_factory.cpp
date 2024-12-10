#include "decoder_factory.h"

namespace dorado::polisher {

std::unique_ptr<DecoderBase> decoder_factory(const ModelConfig& config) {
    const LabelSchemeType label_scheme_type = parse_label_scheme_type(config.label_scheme_type);
    return std::make_unique<DecoderBase>(label_scheme_type);
}

}  // namespace dorado::polisher
