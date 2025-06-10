add_library(dorado_nn STATIC
    # Public
    include/nn/ConvStack.h
    include/nn/CRFModules.h
    include/nn/KoiUtils.h
    include/nn/LinearUpsample.h
    include/nn/LSTMStack.h
    include/nn/MetalModules.h
    include/nn/TxModules.h
    include/nn/WorkingMemory.h

    # Private
    ConvStack.cpp
    CRFModules.cpp
    LinearUpsample.cpp
    LSTMStack.cpp
    TxModules.cpp
    WorkingMemory.cpp
)

if (APPLE)
    target_sources(dorado_nn PRIVATE MetalModules.cpp)
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

target_include_directories(dorado_nn
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_warnings_as_errors(dorado_nn)

if (DORADO_ENABLE_PCH)
    target_precompile_headers(dorado_nn REUSE_FROM dorado_pch)
endif()
