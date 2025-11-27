dorado_add_library(
    NAME
        dorado_nn
    PUBLIC_DIR
        nn
    SOURCES_PUBLIC
        AuxiliaryData.h
        ConvStack.h
        CRFModules.h
        FLSTMStack.h
        KoiThreads.h
        KoiUtils.h
        LinearUpsample.h
        LSTMStack.h
        MetalModules.h
        RNNStack.h
        TxModules.h
        WorkingMemory.h
    SOURCES_PRIVATE
        AuxiliaryData.cpp
        ConvStack.cpp
        CRFModules.cpp
        FLSTMStack.cpp
        LinearUpsample.cpp
        LSTMStack.cpp
        TxModules.cpp
        WorkingMemory.cpp
    DEPENDS_PUBLIC
        dorado_config
        dorado_torch_utils
        dorado_utils
        torch_lib
    DEPENDS_PRIVATE
        nvtx
        spdlog::spdlog
)

if (APPLE)
    target_sources(dorado_nn PRIVATE MetalModules.cpp)
else()
    target_sources(dorado_nn PRIVATE KoiUtils.cpp KoiThreads.cpp)
endif()

if (TARGET koi)
    target_link_libraries(dorado_nn PRIVATE koi)
endif()
