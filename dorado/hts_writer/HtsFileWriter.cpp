#include "hts_writer/HtsFileWriter.h"

namespace dorado {
namespace hts_writer {

void HtsFileWriter::process(const Processable item) {
    // Type-specific dispatch to handle(T)
    dispatch_processable(item, [this](const auto &t) { this->handle(t); });
}

}  // namespace hts_writer
}  // namespace dorado
