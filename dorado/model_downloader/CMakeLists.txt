add_library(dorado_model_downloader_lib STATIC
    downloader.h
    downloader.cpp
    model_downloader.h
    model_downloader.cpp
)

if (APPLE)
    target_sources(dorado_model_downloader_lib PRIVATE downloader_foundation.mm)
endif()

target_compile_definitions(dorado_model_downloader_lib
    PRIVATE
        DORADO_MODELS_HAS_FOUNDATION=$<BOOL:${APPLE}>
        DORADO_MODELS_HAS_HTTPLIB=1
        DORADO_MODELS_HAS_CURL_EXE=1
)

target_link_libraries(dorado_model_downloader_lib
    PUBLIC
        dorado_models_lib
    PRIVATE
        dorado_utils
        elzip
        httplib
        OpenSSL::SSL
        spdlog::spdlog
)

enable_warnings_as_errors(dorado_model_downloader_lib)
