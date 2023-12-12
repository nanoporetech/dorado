set(LIB_SOURCE_FILES
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

add_library(dorado_modbase STATIC ${LIB_SOURCE_FILES})

target_compile_definitions(dorado_modbase
    PUBLIC
        DORADO_GPU_BUILD=$<BOOL:${DORADO_GPU_BUILD}>
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
