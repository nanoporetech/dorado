add_library(dorado_models_lib STATIC
    kits.cpp
    kits.h
    metadata.cpp
    metadata.h
    model_downloader.h
    model_downloader.cpp
    models.cpp
    models.h
)

target_link_libraries(dorado_models_lib
    PRIVATE
        dorado_utils
        OpenSSL::SSL
        elzip
        spdlog::spdlog
)

target_include_directories(dorado_models_lib
    SYSTEM
    PRIVATE
        ${DORADO_3RD_PARTY_SOURCE}/cpp-httplib
)

enable_warnings_as_errors(dorado_models_lib)
