add_library(dorado_config STATIC
    BatchParams.cpp
    BatchParams.h
    BasecallModelConfig.cpp
    BasecallModelConfig.h
    common.cpp
    common.h
    ModBaseBatchParams.cpp
    ModBaseBatchParams.h
    ModBaseModelConfig.cpp
    ModBaseModelConfig.h
)

target_link_libraries(dorado_config
    PUBLIC
        dorado_models_lib
        toml11::toml11
    PRIVATE
        dorado_utils
        spdlog::spdlog
)

enable_warnings_as_errors(dorado_config)
