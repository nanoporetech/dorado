add_library(dorado_splitter STATIC
    # Public
    include/splitter/DuplexReadSplitter.h
    include/splitter/ReadSplitter.h
    include/splitter/RNAReadSplitter.h
    include/splitter/myers.h # only public for unit tests

    # Private
    DuplexReadSplitter.cpp
    myers.cpp
    RNAReadSplitter.cpp
    splitter_utils.cpp
    splitter_utils.h
)

target_link_libraries(dorado_splitter
    PUBLIC
        dorado_utils
    PRIVATE
        dorado_read_pipeline
        spdlog::spdlog
)

target_include_directories(dorado_splitter
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_warnings_as_errors(dorado_splitter)
