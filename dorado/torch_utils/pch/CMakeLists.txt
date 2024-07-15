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
        torch_lib
)

enable_warnings_as_errors(dorado_pch)
