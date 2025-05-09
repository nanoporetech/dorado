cmake_minimum_required(VERSION 3.10)
project(kadayashi CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_KADAYASHI_EXE "Build the kadayashi executable" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(cmake/Warnings.cmake)

find_package(htslib REQUIRED)
find_package(ZLIB REQUIRED)

# Create a static library from the source files
add_library(haplotag_lib STATIC
    src/haplotag_lib/local_haplotagging.cpp
    src/haplotag_lib/types.cpp
    include/kvec.h
    include/local_haplotagging.h
    include/types.h
)

set_target_properties(haplotag_lib PROPERTIES LINKER_LANGUAGE CXX)

add_library(kadayashi::haplotag_lib ALIAS haplotag_lib)

target_include_directories(haplotag_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(haplotag_lib
    PRIVATE
        ${HTSLIB_LIBRARIES}
        ${ZLIB_LIBRARIES}
)

# Optional install
install(TARGETS haplotag_lib EXPORT kadayashi_targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

enable_warnings_as_errors(haplotag_lib)

install(DIRECTORY include/ DESTINATION include)

install(EXPORT kadayashi_targets
    FILE kadayashi.cmake
    NAMESPACE kadayashi::
    DESTINATION lib/cmake/kadayashi
)

# Add the main executable
if (BUILD_KADAYASHI_EXE)
    find_package(Threads REQUIRED)

    add_executable(kadayashi
        src/bam_tagging.cpp
        src/kthread.cpp
        src/cli.cpp
        src/kadayashi_utils.cpp
        src/pipeline.cpp
        src/resources.cpp
        src/main.cpp
    )

    set_target_properties(kadayashi PROPERTIES LINKER_LANGUAGE CXX)

    target_link_libraries(kadayashi
        PRIVATE
            haplotag_lib
            Threads::Threads
            ${HTSLIB_LIBRARIES}
            ${ZLIB_LIBRARIES}
    )
endif()
