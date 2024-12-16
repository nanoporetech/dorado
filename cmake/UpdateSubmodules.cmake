option(GIT_SUBMODULE "Check submodules during build" ON)

function(git_submodule_update)
    find_package(Git QUIET)
    if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")

        if(GIT_SUBMODULE)
            message(STATUS "Submodule update")

            file(LOCK ${CMAKE_SOURCE_DIR} DIRECTORY)
            execute_process(
                COMMAND
                    ${GIT_EXECUTABLE} submodule update
                        --init
                        --recursive
                        # TODO: once we drop centos support we can use this
                        #--depth 1
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT
            )
            file(LOCK ${CMAKE_SOURCE_DIR} DIRECTORY RELEASE)

            if(NOT GIT_SUBMOD_RESULT EQUAL "0")
                message(FATAL_ERROR "git submodule update failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
            endif()
        endif()

    endif()
endfunction()

git_submodule_update()
