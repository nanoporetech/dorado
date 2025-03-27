#include "decoder_factory.h"

#include "secondary/architectures/model_config.h"

namespace dorado::secondary {

std::unique_ptr<DecoderBase> decoder_factory(const ModelConfig& config) {
    const LabelSchemeType label_scheme_type = parse_label_scheme_type(config.label_scheme_type);
    return std::make_unique<DecoderBase>(label_scheme_type);
}

}  // namespace dorado::secondary
