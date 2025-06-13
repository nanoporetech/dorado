dorado_add_library(
    NAME
        dorado_model_downloader_lib
    PUBLIC_DIR
        model_downloader
    SOURCES_PUBLIC
        model_downloader.h
    SOURCES_PRIVATE
        downloader.cpp
        downloader.h
        model_downloader.cpp
    DEPENDS_PUBLIC
        dorado_models_lib
    DEPENDS_PRIVATE
        dorado_utils
        elzip
        httplib
        OpenSSL::SSL
        spdlog::spdlog
    NO_TORCH
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
