set(PUBLIC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/include/secondary/consensus)

target_sources(dorado_secondary
    PUBLIC
        ${PUBLIC_BASE}/consensus_result.h
        ${PUBLIC_BASE}/consensus_utils.h
        ${PUBLIC_BASE}/sample.h
        ${PUBLIC_BASE}/sample_collate_utils.h
        ${PUBLIC_BASE}/sample_trimming.h
        ${PUBLIC_BASE}/variant_calling_sample.h
        ${PUBLIC_BASE}/variant_calling.h
        ${PUBLIC_BASE}/window.h
    PRIVATE
        consensus_utils.cpp
        sample.cpp
        sample_collate_utils.cpp
        sample_trimming.cpp
        variant_calling_sample.cpp
        variant_calling.cpp
        window.cpp
)

target_include_directories(dorado_secondary
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)