# Helper to convert a SBOM YAML into a header full of licences.
#
# Example usage:
#
#   dorado_generate_licence_header_from_yaml(
#       TARGET
#           dorado_licences
#       PATH
#           ${PROJECT_SOURCE_DIR}/dorado/3rdparty
#       SBOM
#           software_versions.yml
#   )
#
function(dorado_generate_licence_header_from_yaml)
    # Parse the args.
    set(options)
    set(oneValueArgs TARGET PATH SBOM)
    set(multiValueArgs)
    cmake_parse_arguments(arg "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(generated_path "${CMAKE_BINARY_DIR}/generated_licences")

    # Create a target that other libs can link to to find the generated header.
    add_library(${arg_TARGET} INTERFACE)
    target_include_directories(${arg_TARGET} INTERFACE "${generated_path}")

    # Start writing the output.
    set(output_file "${generated_path}/${arg_TARGET}/licences.h")
    file(WRITE "${output_file}" "#pragma once\n")
    file(APPEND "${output_file}" "#include <string_view>\n")
    file(APPEND "${output_file}" "namespace ${arg_TARGET} {\n")
    file(APPEND "${output_file}" "inline constexpr struct { std::string_view name, licence; } licences[] = {\n")

    # Read the yaml.
    set(yaml_file "${arg_PATH}/${arg_SBOM}")
    file(STRINGS "${yaml_file}" yaml_lines NO_HEX_CONVERSION)

    # Parse the YAML line by line, assuming that licence is a single line.
    set(dep_name "<not set>")
    set(dep_license "PATH_NOT_AVAILABLE")
    set(dep_omit YES)
    foreach(line IN LISTS yaml_lines)
        if (line MATCHES "^([a-zA-Z].*):$") # new dependency
            set (_dep "${CMAKE_MATCH_1}")
            dorado_emit_licence_for_dependency_("${output_file}" "${dep_name}" "${dep_license}" "${dep_omit}")

            # Setup next dependency.
            set(dep_name "${_dep}")
            set(dep_license "PATH_NOT_AVAILABLE")
            set(dep_omit NO)

        elseif (line MATCHES "^[ \t]+(license|omit):[ \t]+(.*)$") # key-value pair
            set("dep_${CMAKE_MATCH_1}" "${CMAKE_MATCH_2}")
        endif()
    endforeach()
    # Emit the final one.
    dorado_emit_licence_for_dependency_("${output_file}" "${dep_name}" "${dep_license}" "${dep_omit}")

    # Finish off the output.
    file(APPEND "${output_file}" "};\n")
    file(APPEND "${output_file}" "} // namespace ${arg_TARGET}\n")
endfunction()

function(dorado_emit_licence_for_dependency_ OUTPUT NAME LICENCE OMIT)
    if (OMIT)
        return()
    endif()

    if (LICENCE STREQUAL "PATH_NOT_AVAILABLE")
        message(WARNING "No licence file provided for ${NAME} in ${yaml_file}")
        return()
    endif()

    # Look for special cases prefixes.
    set(prefix_pod5 "${DORADO_3RD_PARTY_DOWNLOAD}/${POD5_DIR}")
    if (LICENCE MATCHES "^<(.*)>(.*)$")
        set(prefix prefix_${CMAKE_MATCH_1})
        if (NOT DEFINED ${prefix})
            message(FATAL_ERROR "Unknown prefix: '${CMAKE_MATCH_1}'")
        endif()
        set(licence_path "${${prefix}}/${CMAKE_MATCH_2}")
    else()
        set(licence_path "${arg_PATH}/${LICENCE}")
    endif()

    # Check that it exists.
    if (NOT EXISTS "${licence_path}")
        message(FATAL_ERROR "Missing licence file for ${NAME} in ${yaml_file}: ${licence_path}")
    endif()

    # Add the licence to the header.
    file(READ "${licence_path}" licence_data)
    file(APPEND "${OUTPUT}" "{ \"${NAME}\", ")
    # MSVC can't handle literals longer than ~16K so we split up the strings. See error C2026.
    set(substr_length 16000)
    string(LENGTH "${licence_data}" licence_length)
    foreach(substr_start RANGE 0 ${licence_length} ${substr_length})
        string(SUBSTRING "${licence_data}" ${substr_start} ${substr_length} licence_substr)
        file(APPEND "${OUTPUT}" "R\"licence_delim(${licence_substr})licence_delim\" ")
    endforeach()
    file(APPEND "${OUTPUT}" " },\n")
endfunction()


# Third party licences.
dorado_generate_licence_header_from_yaml(
    TARGET dorado_licences
    PATH ${PROJECT_SOURCE_DIR}/dorado/3rdparty
    SBOM software_versions.yml
)

# POD5 licences.
dorado_generate_licence_header_from_yaml(
    TARGET dorado_licences_pod5
    PATH ${PROJECT_SOURCE_DIR}/dorado/3rdparty
    SBOM software_versions_pod5.yml
)
