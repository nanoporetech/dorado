add_library(dorado_secondary_consensus
    consensus_result.h
    consensus_utils.cpp
    consensus_utils.h
    sample.cpp
    sample.h
    sample_trimming.cpp
    sample_trimming.h
    variant_calling.cpp
    variant_calling.h
    variant_calling_sample.cpp
    variant_calling_sample.h
    window.cpp
    window.h
)

target_include_directories(dorado_secondary_consensus
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/consensus
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
