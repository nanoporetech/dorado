add_library(dorado_correct STATIC
    # Public
    include/correct/conversions.h
    include/correct/CorrectionProgressTracker.h
    include/correct/decode.h
    include/correct/features.h
    include/correct/infer.h
    include/correct/types.h
    include/correct/windows.h

    # Private
    conversions.cpp
    CorrectionProgressTracker.cpp
    decode.cpp
    features.cpp
    infer.cpp
    types.cpp
    windows.cpp
)

target_link_libraries(dorado_correct
    PUBLIC
        dorado_secondary
        dorado_torch_utils
        dorado_utils
        indicators
        spdlog::spdlog
    PRIVATE
        dorado_read_pipeline_base
        toml11::toml11
)

target_include_directories(dorado_correct
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_warnings_as_errors(dorado_correct)
