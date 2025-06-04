add_library(dorado_models_lib STATIC
    # Public
    include/models/kits.h
    include/models/metadata.h
    include/models/model_complex.h
    include/models/models.h

    # Private
    kits.cpp
    metadata.cpp
    model_complex.cpp
    models.cpp
)

target_link_libraries(dorado_models_lib
    PRIVATE
        dorado_utils
        spdlog::spdlog
)

target_include_directories(dorado_models_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/.. # TODO: remove this once dependents are fixed
)

enable_warnings_as_errors(dorado_models_lib)
