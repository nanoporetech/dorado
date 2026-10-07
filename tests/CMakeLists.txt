option(DORADO_DISABLE_TESTS "Disable building the tests" OFF)
option(DORADO_ENABLE_BENCHMARK_TESTS "Enable benchmarks to run as part of the tests" OFF)

if (DORADO_DISABLE_TESTS)
    return()
endif()

# dorado_tests
add_executable(dorado_tests
    AdapterDetectorTest.cpp
    AlignerTest.cpp
    arg_parse_ext_test.cpp
    AsyncQueueTest.cpp
    BamReaderTest.cpp
    BamUtilsTest.cpp
    BamWriterTest.cpp
    BarcodeClassifierSelectorTest.cpp
    BarcodeClassifierTest.cpp
    BasecallModelConfigTest.cpp
    BasecallModelTest.cpp
    BatchParamsTest.cpp
    bed_file_test.cpp
    ChunkTest.cpp
    CigarTest.cpp
    CliUtilsTest.cpp
    context_container_test.cpp
    CorrectionWindowTest.cpp
    CustomBarcodeParserTest.cpp
    DuplexReadTaggingNodeTest.cpp
    DuplexSplitTest.cpp
    FastqTagsTest.cpp
    FastxRandomReaderTest.cpp
    FastxSequentialReaderTest.cpp
    FileInfoTest.cpp
    FixedSizeQueueTest.cpp
    gpu_monitor_test.cpp
    HeaderMapperTest.cpp
    HtsFileTest.cpp
    IndexFileAccessTest.cpp
    KadayashiTest.cpp
    MathUtilsTest.cpp
    MergeHeadersTest.cpp
    Minimap2IndexTest.cpp
    ModBaseChunkTest.cpp
    ModBaseConfigTest.cpp
    ModBaseContextTest.cpp
    ModBaseEncoderTest.cpp
    ModBaseLatentSpaceV1ModelTest.cpp
    ModelComplexTest.cpp
    ModelKitsTest.cpp
    ModelMetadataTest.cpp
    ModelUtilsTest.cpp
    MotifMatcherTest.cpp
    myers_test.cpp
    PafUtilsTest.cpp
    PairingNodeTest.cpp
    PipelineTest.cpp
    Pod5DataLoaderTest.cpp
    PolyACalculatorTest.cpp
    PostConditionTest.cpp
    ReadFilterNodeTest.cpp
    ReadForwarderNodeTest.cpp
    ReadTest.cpp
    ReadToBamTypeNodeTest.cpp
    RealignMovesTest.cpp
    ResourceLimiterTest.cpp
    ResumeLoaderTest.cpp
    RleTest.cpp
    RNASplitTest.cpp
    SampleSheetTests.cpp
    SamUtilsTest.cpp
    ScaledDotProductAttention.cpp
    SecondaryBamFileTest.cpp
    SecondaryBamInfoTest.cpp
    SecondaryDecodeVariantsTest.cpp
    SecondaryEncoderReadAlignmentTest.cpp
    SecondaryEncoderUtilsTest.cpp
    SecondaryKadayashiUtils.cpp
    SecondaryModelConfigValidation.cpp
    SecondaryVCSamplesTest.cpp
    SecondaryModelFactory.cpp
    SecondaryNormalizeVariantsTest.cpp
    SecondaryModelVariantPerceiverTest.cpp
    SecondarySampleCollateUtilsTest.cpp
    SecondarySampleTest.cpp
    SecondaryTrimTest.cpp
    SecondaryWindowTest.cpp
    SeparatedStreamTest.cpp
    SequenceUtilsTest.cpp
    StereoDuplexTest.cpp
    StitchTest.cpp
    StlUtilsTest.cpp
    StringUtilsTest.cpp
    SummaryFileWriterTest.cpp
    synchronisation_test.cpp
    TimeUtilsTest.cpp
    TrimTest.cpp
    VCFWriterTest.cpp
    VariantImplTest.cpp
    WriterNodeTest.cpp
)

if(APPLE)
    target_sources(dorado_tests PRIVATE MetalLinearTest.cpp)
else()
    target_sources(dorado_tests PRIVATE cuda_utils_test.cpp)
endif()

# dorado_smoke_tests
add_executable(dorado_smoke_tests
    NodeSmokeTest.cpp
)


# dorado_tests_common
add_library(dorado_tests_common STATIC
    main.cpp
    TestUtils.cpp
)
enable_warnings_as_errors(dorado_tests_common)
target_include_directories(dorado_tests_common
    PUBLIC
        # Per-target tests need to be able to include TestUtils.h
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(dorado_tests_common
    PUBLIC
        Catch2::Catch2
    PRIVATE
        # Note: these are only needed for the setup in main()
        # TODO: try and remove the torch dependency so that dorado_utils' tests doesn't link to it
        dorado_torch_utils
        dorado_utils
)
target_compile_definitions(dorado_tests_common
    PUBLIC
        DORADO_ENABLE_BENCHMARK_TESTS=$<BOOL:${DORADO_ENABLE_BENCHMARK_TESTS}>
        # Workaround torch defining a CATCH_CHECK() macro that conflicts with catch2
        CATCH_CONFIG_PREFIX_ALL=1
)


# Finish setting up each target and add them as tests.
foreach(TEST_BIN dorado_tests dorado_smoke_tests)
    if (DORADO_ENABLE_PCH)
        target_precompile_headers(${TEST_BIN} REUSE_FROM dorado_pch_exe)
    endif()

    target_link_libraries(${TEST_BIN}
        PRIVATE
            dorado_basecall
            dorado_cli
            dorado_file_info
            dorado_io_lib
            dorado_lib
            dorado_modbase
            dorado_model_resolver_lib
            dorado_model_downloader_lib
            dorado_models_lib
            dorado_nn
            dorado_polish
            dorado_secondary
            dorado_tests_common
            dorado_variant
            intervaltree
            minimap2
            toml11::toml11
            ${ZLIB_LIBRARIES}
    )

    enable_warnings_as_errors(${TEST_BIN})
    if (MSVC)
        set_property(TARGET ${TEST_BIN} APPEND PROPERTY LINK_OPTIONS "/ignore:4099")
    else()
        # Add -Wno-trigraphs because one of the tests uses a qstring that has a trigraph sequence.
        target_compile_options(${TEST_BIN} PRIVATE "-Wno-trigraphs")
    endif()

    # Add the test.
    dorado_add_test(${TEST_BIN})
endforeach()

# GCC 8 ICEs trying to compile this file with ASAN+optimisations enabled, so knock down the optimisation to try and help it out.
if (ECM_ENABLE_SANITIZERS AND (CMAKE_CXX_COMPILER_ID MATCHES "GNU") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0))
    set_source_files_properties(TrimTest.cpp PROPERTIES COMPILE_OPTIONS "-O0")
endif()
