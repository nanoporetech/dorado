add_library(dorado_io_lib
    # Public
    include/data_loader/DataLoader.h

    # Private
    DataLoader.cpp
)

target_link_libraries(dorado_io_lib
    PUBLIC
        cxxpool
        dorado_lib
    PRIVATE
        pod5_libs
        spdlog::spdlog
)

target_include_directories(dorado_io_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_warnings_as_errors(dorado_io_lib)
