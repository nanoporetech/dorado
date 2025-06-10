add_library(dorado_polish STATIC
    # Public
    include/polish/polish_impl.h

    # Private
    polish_impl.cpp
)

target_link_libraries(dorado_polish
    PUBLIC
        dorado_secondary
        dorado_utils
    PRIVATE
        cxxpool
        dorado_hts_io
        dorado_torch_utils
        htslib
        spdlog::spdlog
)

target_include_directories(dorado_polish
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_warnings_as_errors(dorado_polish)
