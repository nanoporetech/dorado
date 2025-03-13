add_library(dorado_config STATIC
    BatchParams.cpp
    BatchParams.h
    BasecallModelConfig.cpp
    BasecallModelConfig.h
    ModBaseBatchParams.cpp
    ModBaseBatchParams.h
    ModBaseModelConfig.cpp
    ModBaseModelConfig.h
)

target_link_libraries(dorado_config
    PUBLIC
        dorado_models_lib
    PRIVATE
        dorado_utils
        spdlog::spdlog
        toml11::toml11
)

enable_warnings_as_errors(dorado_config)
