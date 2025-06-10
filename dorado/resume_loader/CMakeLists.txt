add_library(dorado_resume_loader STATIC
    # Public
    include/resume_loader/ResumeLoader.h

    # Private
    ResumeLoader.cpp
)

target_link_libraries(dorado_resume_loader
    PUBLIC
        dorado_read_pipeline_base
    PRIVATE
        dorado_utils
        htslib
        indicators
        spdlog::spdlog
)

target_include_directories(dorado_resume_loader
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_warnings_as_errors(dorado_resume_loader)
