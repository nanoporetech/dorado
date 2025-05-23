add_library(dorado_nn STATIC
    ConvStack.cpp
    ConvStack.h
    CRFModules.cpp
    CRFModules.h
    KoiUtils.cpp
    KoiUtils.h
    LinearUpsample.cpp
    LinearUpsample.h
    LSTMStack.cpp
    LSTMStack.h
    TxModules.cpp
    TxModules.h
    WorkingMemory.cpp
    WorkingMemory.h
)


if (DORADO_GPU_BUILD AND APPLE)
    target_sources(dorado_nn
        PRIVATE
        metal/MetalModules.cpp
        metal/MetalModules.h
    )
endif()

target_include_directories(dorado_nn
    SYSTEM
    PRIVATE
        ${DORADO_3RD_PARTY_SOURCE}/NVTX/c/include
)

target_link_libraries(dorado_nn
    PUBLIC
        torch_lib
        dorado_torch_utils
        dorado_utils
        dorado_config
    PRIVATE
        spdlog::spdlog
)

if (TARGET koi)
    target_link_libraries(dorado_nn PRIVATE koi)
endif()

enable_warnings_as_errors(dorado_nn)

if (DORADO_ENABLE_PCH)
    target_precompile_headers(dorado_nn REUSE_FROM dorado_torch_utils)
endif()
