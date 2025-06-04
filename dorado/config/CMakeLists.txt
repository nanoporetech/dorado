add_library(dorado_config STATIC
    # Public
    include/config/BatchParams.h
    include/config/BasecallModelConfig.h
    include/config/common.h
    include/config/ModBaseBatchParams.h
    include/config/ModBaseModelConfig.h

    # Private
    BatchParams.cpp
    BasecallModelConfig.cpp
    common.cpp
    ModBaseBatchParams.cpp
    ModBaseModelConfig.cpp
)

target_link_libraries(dorado_config
    PUBLIC
        dorado_models_lib
    PRIVATE
        dorado_utils
        spdlog::spdlog
        toml11::toml11
)

target_include_directories(dorado_config
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_warnings_as_errors(dorado_config)
