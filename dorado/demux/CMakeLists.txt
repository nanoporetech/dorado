add_library(dorado_demux STATIC
    # Public
    include/demux/adapter_info.h
    include/demux/AdapterDetector.h
    include/demux/AdapterDetectorSelector.h
    include/demux/BarcodeClassifier.h
    include/demux/BarcodeClassifierSelector.h
    include/demux/barcoding_info.h
    include/demux/KitInfoProvider.h
    include/demux/parse_custom_kit.h
    include/demux/Trimmer.h

    # Private
    adapter_primer_kits.cpp
    adapter_primer_kits.h
    AdapterDetector.cpp
    AdapterDetectorSelector.cpp
    BarcodeClassifier.cpp
    BarcodeClassifierSelector.cpp
    KitInfoProvider.cpp
    parse_custom_kit.cpp
    parse_custom_sequences.cpp
    Trimmer.cpp
)

target_link_libraries(dorado_demux
    PUBLIC
        dorado_utils
    PRIVATE
        dorado_read_pipeline
        htslib
        spdlog::spdlog
        toml11::toml11
)

target_include_directories(dorado_demux
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_warnings_as_errors(dorado_demux)
