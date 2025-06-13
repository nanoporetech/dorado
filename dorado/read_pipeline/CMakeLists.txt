add_subdirectory(base)
add_subdirectory(nodes)

add_library(dorado_read_pipeline INTERFACE)
target_link_libraries(dorado_read_pipeline
    INTERFACE
        dorado_read_pipeline_base
        dorado_read_pipeline_nodes
)
