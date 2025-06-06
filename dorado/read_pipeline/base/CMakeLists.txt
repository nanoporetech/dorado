add_library(dorado_read_pipeline_base STATIC
    # Public
    include/read_pipeline/base/ClientInfo.h
    include/read_pipeline/base/context_container.h
    include/read_pipeline/base/DefaultClientInfo.h
    include/read_pipeline/base/flush_options.h
    include/read_pipeline/base/messages.h
    include/read_pipeline/base/MessageSink.h
    include/read_pipeline/base/read_utils.h
    include/read_pipeline/base/ReadPipeline.h

    # Private
    messages.cpp
    MessageSink.cpp
    read_utils.cpp
    ReadPipeline.cpp
    stereo_features.cpp
    stereo_features.h
)

target_link_libraries(dorado_read_pipeline_base
    PUBLIC
        dorado_models_lib
        dorado_torch_utils
        dorado_utils
    PRIVATE
        dorado_modbase
        htslib
        spdlog::spdlog
)

target_include_directories(dorado_read_pipeline_base
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_warnings_as_errors(dorado_read_pipeline_base)

# GCC 8 ICEs trying to compile this file with ASAN+optimisations enabled, so knock down the optimisation to try and help it out.
if (ECM_ENABLE_SANITIZERS AND (CMAKE_CXX_COMPILER_ID MATCHES "GNU") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0))
    set_source_files_properties(stereo_features.cpp PROPERTIES COMPILE_OPTIONS "-O0")
endif()
