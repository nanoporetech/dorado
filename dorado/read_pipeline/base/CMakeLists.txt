dorado_add_library(
    NAME
        dorado_read_pipeline_base
    PUBLIC_DIR
        read_pipeline/base
    SOURCES_PUBLIC
        ClientInfo.h
        DefaultClientInfo.h
        flush_options.h
        HtsReader.h
        messages.h
        MessageSink.h
        read_utils.h
        ReadPipeline.h
        stitch.h
    SOURCES_PRIVATE
        HtsReader.cpp
        messages.cpp
        MessageSink.cpp
        read_utils.cpp
        ReadPipeline.cpp
        stereo_features.cpp
        stereo_features.h
        stitch.cpp
    DEPENDS_PUBLIC
        dorado_models_lib
        dorado_torch_utils
        dorado_utils
        htslib
    DEPENDS_PRIVATE
        dorado_modbase
        spdlog::spdlog
)

# GCC 8 ICEs trying to compile this file with ASAN+optimisations enabled, so knock down the optimisation to try and help it out.
if (ECM_ENABLE_SANITIZERS AND (CMAKE_CXX_COMPILER_ID MATCHES "GNU") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0))
    set_source_files_properties(stereo_features.cpp PROPERTIES COMPILE_OPTIONS "-O0")
endif()
