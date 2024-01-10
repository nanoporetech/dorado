#include "messages.h"

namespace dorado {

bool is_read_message(const Message &message) {
    return std::holds_alternative<SimplexReadPtr>(message) ||
           std::holds_alternative<DuplexReadPtr>(message);
}

uint64_t SimplexRead::get_end_time_ms() const {
    return read_common.start_time_ms +
           ((end_sample - start_sample) * 1000) /
                   read_common.sample_rate;  //TODO get rid of the trimmed thing?
}

}  // namespace dorado