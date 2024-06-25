add_library(dorado_models_lib STATIC
    kits.cpp
    kits.h
    metadata.cpp
    metadata.h
    model_complex.h
    model_complex.cpp
    models.cpp
    models.h
)

target_link_libraries(dorado_models_lib
    PRIVATE
        dorado_utils
        spdlog::spdlog
)

enable_warnings_as_errors(dorado_models_lib)
