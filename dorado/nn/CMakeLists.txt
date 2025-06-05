add_library(dorado_nn STATIC
    ConvStack.cpp
    ConvStack.h
    CRFModules.cpp
    CRFModules.h
    KoiUtils.h
    LinearUpsample.cpp
    LinearUpsample.h
    LSTMStack.cpp
    LSTMStack.h
    TxModules.cpp
    TxModules.h
    WorkingMemory.cpp
    WorkingMemory.h
    metal/MetalModules.h
)

if (APPLE)
    target_sources(dorado_nn PRIVATE metal/MetalModules.cpp)
else()
    target_sources(dorado_nn PRIVATE KoiUtils.cpp)
endif()

target_link_libraries(dorado_nn
    PUBLIC
        dorado_config
        dorado_torch_utils
        dorado_utils
        torch_lib
    PRIVATE
        nvtx
        spdlog::spdlog
)

if (TARGET koi)
    target_link_libraries(dorado_nn PRIVATE koi)
endif()

enable_warnings_as_errors(dorado_nn)

if (DORADO_ENABLE_PCH)
    target_precompile_headers(dorado_nn REUSE_FROM dorado_torch_utils)
endif()
