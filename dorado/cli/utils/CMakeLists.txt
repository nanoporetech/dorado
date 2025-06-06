add_library(dorado_cli_utils INTERFACE)
target_include_directories(dorado_cli_utils
    INTERFACE
        include
)
target_link_libraries(dorado_cli_utils
    INTERFACE
        dorado_version
        dorado_utils
        htslib
        dorado_torch_utils
)
