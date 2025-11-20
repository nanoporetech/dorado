cmake_minimum_required(VERSION 3.10)
project(kadayashi CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



option(BUILD_KADAYASHI_ENABLE_COVERAGE "Build tests with gcov coverage" OFF)
option(BUILD_KADAYASHI_TESTS "Build the kadayashi unit test executable" ON)
option(BUILD_KADAYASHI_EXE "Build the kadayashi executable" ON)
option(BUILD_KADAYASHI_AS_SUBPROJECT "If ON, custom compiler options will not be set" OFF)

# Helper: enable gcov only for selected targets
function(enable_gcov target)
  if(BUILD_KADAYASHI_ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${target} PRIVATE -O0 -g -fprofile-arcs -ftest-coverage --coverage)
    target_link_options(${target} PRIVATE -lgcov --coverage)
  endif()
endfunction()

# Helper: -Wall
function(enable_wall target)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${target} PRIVATE -Wall)
  endif()
endfunction()

# If no prefix is provided we install next to the binary directory.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/../dist CACHE PATH "" FORCE)
endif()

message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Set custom compiler options only if not inherited from a top level project.
if (NOT BUILD_KADAYASHI_AS_SUBPROJECT)
    include(cmake/Warnings.cmake)
endif()

find_package(htslib REQUIRED)
find_package(ZLIB REQUIRED)

################################
### Create the haplotag_lib. ###
################################
# Create a static library from the source files
add_library(haplotag_lib STATIC
    src/haplotag_lib/local_haplotagging.cpp
    src/haplotag_lib/hts_types.cpp
    src/haplotag_lib/blocked_bloom_filter.cpp
    src/haplotag_lib/faidx_utils.cpp
    src/haplotag_lib/sequence_utility.cpp
    src/haplotag_lib/variant_graph.cpp
    src/haplotag_lib/bam_record_parsing.cpp
    include/bam_file_view.h
    include/local_haplotagging.h
    include/faidx_utils.h
    include/hts_types.h
    include/types.h
    include/blocked_bloom_filter.h
)


set_target_properties(haplotag_lib PROPERTIES LINKER_LANGUAGE CXX)

add_library(kadayashi::haplotag_lib ALIAS haplotag_lib)

target_include_directories(haplotag_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/hts_utils>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/haplotag_lib
)

target_link_libraries(haplotag_lib
    PRIVATE
        ${HTSLIB_LIBRARIES}
        ${ZLIB_LIBRARIES}
)
enable_gcov(haplotag_lib)
enable_wall(haplotag_lib)

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

###########################################
### Create the lib for everything else. ###
###########################################
if (BUILD_KADAYASHI_EXE OR BUILD_KADAYASHI_TESTS)
    add_library(kadayashi_main_lib STATIC
            src/bam_tagging.cpp
            src/kthread.cpp
            src/cli.cpp
            src/kadayashi_utils.cpp
            src/pipeline.cpp
            src/resources.cpp
            src/hts_utils/BamFile.cpp
            src/hts_utils/FastxRandomReader.cpp
    )
    target_include_directories(kadayashi_main_lib
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/haplotag_lib
        ${CMAKE_CURRENT_SOURCE_DIR}/src/hts_utils
    )
    target_link_libraries(kadayashi_main_lib
        PRIVATE
            haplotag_lib
            Threads::Threads
            ${HTSLIB_LIBRARIES}
            ${ZLIB_LIBRARIES}
    )
    enable_gcov(kadayashi_main_lib)
    enable_wall(kadayashi_main_lib)
endif()
###########################################

if (BUILD_KADAYASHI_TESTS)
    add_subdirectory(tests)
endif()

# Add the main executable
if (BUILD_KADAYASHI_EXE)
    find_package(Threads REQUIRED)

    add_executable(kadayashi
        src/main.cpp
    )

    set_target_properties(kadayashi PROPERTIES LINKER_LANGUAGE CXX)

    target_link_libraries(kadayashi
        PRIVATE
            haplotag_lib
            kadayashi_main_lib
            ${HTSLIB_LIBRARIES}
            ${ZLIB_LIBRARIES}
    )
    enable_gcov(kadayashi)
    enable_wall(kadayashi)

    install(TARGETS kadayashi DESTINATION bin)

endif()
