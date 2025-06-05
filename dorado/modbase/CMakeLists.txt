add_library(dorado_modbase STATIC
    encode_kmer.cpp
    encode_kmer.h
    ModBaseCaller.cpp
    ModBaseCaller.h
    ModBaseContext.cpp
    ModBaseContext.h
    ModbaseEncoder.cpp
    ModbaseEncoder.h
    ModBaseRunner.cpp
    ModBaseRunner.h
    ModbaseScaler.cpp
    ModbaseScaler.h
    MotifMatcher.cpp
    MotifMatcher.h
    nn/ModBaseModel.cpp
    nn/ModBaseModel.h
)

target_link_libraries(dorado_modbase
    PUBLIC
        dorado_config
        torch_lib
    PRIVATE
        dorado_torch_utils
        dorado_utils
        spdlog::spdlog
)

enable_warnings_as_errors(dorado_modbase)

if (DORADO_ENABLE_PCH)
    target_precompile_headers(dorado_modbase REUSE_FROM dorado_torch_utils)
endif()
