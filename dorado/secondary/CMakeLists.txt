add_subdirectory(architectures)
add_subdirectory(consensus)
add_subdirectory(features)
add_subdirectory(common)

add_library(dorado_secondary INTERFACE)
target_link_libraries(dorado_secondary
    INTERFACE
        dorado_secondary_architectures
        dorado_secondary_common
        dorado_secondary_consensus
        dorado_secondary_features
)
