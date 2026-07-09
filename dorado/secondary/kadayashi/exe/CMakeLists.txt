if (BUILD_KADAYASHI_EXE)
    find_package(Threads REQUIRED)
    add_executable(kadayashi
        ${CMAKE_CURRENT_SOURCE_DIR}/bam_tagging.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cli.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/kadayashi_utils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/pipeline.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/resources.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    )

    target_link_libraries(kadayashi
        PUBLIC
            dorado_hts_utils
            dorado_secondary
        PRIVATE
            cxxpool
            Threads::Threads
            htslib
            ${ZLIB_LIBRARIES}
            spdlog::spdlog
            klib
    )

    install(TARGETS kadayashi DESTINATION bin)

endif()
