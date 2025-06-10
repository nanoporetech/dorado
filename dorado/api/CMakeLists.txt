add_library(dorado_api STATIC
    # Public
    include/api/caller_creation.h
    include/api/pipeline_creation.h
    include/api/runner_creation.h

    # Private
    caller_creation.cpp
    pipeline_creation.cpp
    runner_creation.cpp
)

target_link_libraries(dorado_api
    PUBLIC
        dorado_basecall
        dorado_modbase
        dorado_read_pipeline_base
    PRIVATE
        cxxpool
        dorado_read_pipeline_nodes
        dorado_splitter
        dorado_utils
        spdlog::spdlog
)

target_include_directories(dorado_api
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_warnings_as_errors(dorado_api)
