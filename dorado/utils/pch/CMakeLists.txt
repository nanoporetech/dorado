add_library(dorado_pch
    precompiled.cpp
    precompiled.h
)

target_precompile_headers(dorado_pch
    PUBLIC
        precompiled.h
)

target_link_libraries(dorado_pch
    PUBLIC
        ${TORCH_LIBRARIES}
)

# 3rdparty libs should be considered SYSTEM headers
target_include_directories(dorado_pch
    SYSTEM
    PUBLIC
        ${TORCH_INCLUDE_DIRS}
)

enable_warnings_as_errors(dorado_pch)
