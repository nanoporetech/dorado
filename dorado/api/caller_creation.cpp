#include "api/caller_creation.h"

#include "basecall/ModelCallerFactory.h"
#include "modbase/ModBaseCaller.h"

namespace dorado::api {

std::shared_ptr<basecall::ModelCaller> create_basecall_caller(
        const basecall::BasecallerCreationParams& params) {
    return basecall::create_model_caller(params);
}

std::shared_ptr<modbase::ModBaseCaller> create_modbase_caller(
        const std::span<const config::ModBaseModelConfig> model_configs,
        int batch_size,
        const std::string& device) {
    return std::make_shared<modbase::ModBaseCaller>(model_configs, batch_size, device);
}

}  // namespace dorado::api
