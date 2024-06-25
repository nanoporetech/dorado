add_library(dorado_model_downloader_lib STATIC
    downloader.h
    downloader.cpp
    model_downloader.h
    model_downloader.cpp
)


if (APPLE)
    target_sources(dorado_model_downloader_lib PRIVATE downloader_foundation.mm)
endif()
if (NOT IOS)
    target_link_libraries(dorado_model_downloader_lib PRIVATE OpenSSL::SSL)
endif()

target_compile_definitions(dorado_model_downloader_lib
    PRIVATE
        DORADO_MODELS_HAS_FOUNDATION=$<BOOL:${APPLE}>
        DORADO_MODELS_HAS_HTTPLIB=$<NOT:$<BOOL:${IOS}>>
        DORADO_MODELS_HAS_CURL_EXE=$<NOT:$<BOOL:${IOS}>>
)


target_link_libraries(dorado_model_downloader_lib
    PUBLIC
        dorado_models_lib
    PRIVATE
        dorado_utils
        elzip
        spdlog::spdlog
)

target_include_directories(dorado_model_downloader_lib
    SYSTEM
    PRIVATE
        ${DORADO_3RD_PARTY_SOURCE}/cpp-httplib
)

enable_warnings_as_errors(dorado_model_downloader_lib)
