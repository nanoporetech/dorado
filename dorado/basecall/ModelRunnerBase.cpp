#include "ModelRunnerBase.h"

#include "config/BasecallModelConfig.h"

namespace dorado::basecall {
std::pair<int, int> ModelRunnerBase::batch_timeouts_ms() const {
    return config::is_duplex_model(config()) ? std::make_pair(5000, 5000)
                                             : std::make_pair(100, 100);
}
}  // namespace dorado::basecall