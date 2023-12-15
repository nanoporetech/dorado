add_library(dorado_modbase STATIC
    ModBaseContext.cpp
    ModBaseContext.h
    ModbaseEncoder.cpp
    ModbaseEncoder.h
    ModBaseModel.cpp
    ModBaseModel.h
    ModBaseModelConfig.cpp
    ModBaseModelConfig.h    
    ModBaseRunner.cpp
    ModBaseRunner.h
    ModbaseScaler.cpp
    ModbaseScaler.h
    MotifMatcher.cpp
    MotifMatcher.h
)

target_include_directories(dorado_modbase
    SYSTEM
    PUBLIC
    PRIVATE
        ${DORADO_3RD_PARTY_SOURCE}/NVTX/c/include
        ${DORADO_3RD_PARTY_SOURCE}/toml11
)

target_link_libraries(dorado_modbase
    PUBLIC
        ${TORCH_LIBRARIES}
    PRIVATE
        dorado_utils
        spdlog::spdlog
)

enable_warnings_as_errors(dorado_modbase)

if (DORADO_ENABLE_PCH)
    target_precompile_headers(dorado_modbase REUSE_FROM dorado_utils)
endif()
