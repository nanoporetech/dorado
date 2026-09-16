option(BUILD_KADAYASHI_EXE "Build the kadayashi executable" ON)

set(PUBLIC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/include/secondary/kadayashi)
target_sources(dorado_secondary
    PUBLIC
        ${PUBLIC_BASE}/local_haplotagging.h
        ${PUBLIC_BASE}/faidx_utils.h
        ${PUBLIC_BASE}/types.h
    PRIVATE
        local_haplotagging.cpp
        faidx_utils.cpp
        sequence_utility.cpp
        bam_record_parsing.cpp
        variant_graph.cpp
        variant_graph.h
        string_utils.cpp
        string_utils.h
)
target_include_directories(dorado_secondary
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (BUILD_KADAYASHI_EXE)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/exe")
endif()
