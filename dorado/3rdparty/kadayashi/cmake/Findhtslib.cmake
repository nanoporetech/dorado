if (TARGET htslib)
    set(HTSLIB_FOUND TRUE)
    set(HTSLIB_LIBRARIES htslib)
    message(NOTICE "[FindHtslib] Found Htslib through the 'htslib' target")

elseif (TARGET htslib::hts)
    set(HTSLIB_FOUND TRUE)
    set(HTSLIB_LIBRARIES htslib::hts)
    message(NOTICE "[FindHtslib] Found Htslib through the 'htslib::hts' target")

else()
    # Optional path to Htslib (-DHTSLIB_ROOT=/path/to/htslib)
    if (DEFINED HTSLIB_ROOT)
        list(PREPEND CMAKE_PREFIX_PATH "${HTSLIB_ROOT}")
    endif()

    # Try to find HTSLIB via CMake and fallback to pkg-config.
    find_package(htslib QUIET NO_MODULE)

    if (htslib_FOUND)
        set(HTSLIB_FOUND TRUE)
        set(HTSLIB_LIBRARIES htslib::hts)
        message(NOTICE "[FindHtslib] Found htslib via config package")
    else()
        # Fallback to pkg-config
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(HTSLIB REQUIRED htslib)
        include_directories(${HTSLIB_INCLUDE_DIRS})
        link_directories(${HTSLIB_LIBRARY_DIRS})

        set(HTSLIB_FOUND TRUE)
        set(HTSLIB_LIBRARIES ${HTSLIB_LIBRARIES})
        message(NOTICE "[FindHtslib] Found htslib via pkg-config fallback")
    endif()
endif()

mark_as_advanced(HTSLIB_LIBRARIES HTSLIB_FOUND)