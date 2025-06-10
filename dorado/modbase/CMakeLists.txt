add_library(dorado_modbase STATIC
    # Public
    include/modbase/encode_kmer.h
    include/modbase/ModBaseCaller.h
    include/modbase/ModBaseContext.h
    include/modbase/ModBaseEncoder.h
    include/modbase/ModBaseRunner.h
    include/modbase/MotifMatcher.h

    # Private
    encode_kmer.cpp
    ModBaseCaller.cpp
    ModBaseContext.cpp
    ModBaseEncoder.cpp
    ModBaseRunner.cpp
    ModBaseScaler.cpp
    ModBaseScaler.h
    MotifMatcher.cpp
    nn/ModBaseModel.cpp
    nn/ModBaseModel.h
)

target_link_libraries(dorado_modbase
    PUBLIC
        dorado_config
        torch_lib
    PRIVATE
        dorado_nn
        dorado_torch_utils
        dorado_utils
        spdlog::spdlog
)

target_include_directories(dorado_modbase
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_warnings_as_errors(dorado_modbase)

if (DORADO_ENABLE_PCH)
    target_precompile_headers(dorado_modbase REUSE_FROM dorado_pch)
endif()
