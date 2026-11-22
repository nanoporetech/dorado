set(DORADO_VERSION_MAJOR 2)
set(DORADO_VERSION_MINOR 2)
set(DORADO_VERSION_REV 0)

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} log -1 --pretty=format:%h
    OUTPUT_VARIABLE DORADO_SHORT_HASH
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMAND_ERROR_IS_FATAL ANY
  )
  execute_process(
    COMMAND ${GIT_EXECUTABLE} diff --exit-code
    RESULT_VARIABLE IS_DIRTY
    OUTPUT_VARIABLE GIT_DIFF_MSG
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  )
  if(IS_DIRTY)
    # Limit to the first few lines so that local builds don't produce huge logs.
    string(SUBSTRING "${GIT_DIFF_MSG}" 0 250 GIT_DIFF_MSG)
    message(WARNING "Dirty build:\n${GIT_DIFF_MSG}")
    string(APPEND DORADO_SHORT_HASH "+dirty")
  endif()
else()
  set(DORADO_SHORT_HASH 0)
endif()

set(DORADO_VERSION "${DORADO_VERSION_MAJOR}.${DORADO_VERSION_MINOR}.${DORADO_VERSION_REV}+${DORADO_SHORT_HASH}")

# Setup a target that you can link to get version info.
configure_file(
    dorado/dorado_version.h.in
    ${CMAKE_BINARY_DIR}/dorado_version/dorado_version.h
)
add_library(dorado_version INTERFACE)
target_include_directories(dorado_version INTERFACE ${CMAKE_BINARY_DIR}/dorado_version)
