add_library(dorado_config STATIC
    BasecallerParams.cpp
    BasecallerParams.h
    CRFModelConfig.cpp
    CRFModelConfig.h
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
