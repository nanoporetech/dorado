add_library(dorado_secondary_consensus
    # Public
    include/secondary/consensus/consensus_result.h
    include/secondary/consensus/consensus_utils.h
    include/secondary/consensus/sample_trimming.h
    include/secondary/consensus/sample.h
    include/secondary/consensus/variant_calling_sample.h
    include/secondary/consensus/variant_calling.h
    include/secondary/consensus/window.h

    # Private
    consensus_utils.cpp
    sample_trimming.cpp
    sample.cpp
    variant_calling_sample.cpp
    variant_calling.cpp
    window.cpp
)

target_include_directories(dorado_secondary_consensus
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(dorado_secondary_consensus
    PUBLIC
        dorado_torch_utils
        dorado_utils
    PRIVATE
        cxxpool
        spdlog::spdlog
)

enable_warnings_as_errors(dorado_secondary_consensus)
