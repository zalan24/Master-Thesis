function(download directory)
    include(Vindoz)

    get_filename_component(name "${directory}" NAME)
    get_filename_component(directory "${directory}" ABSOLUTE)

    set(source_dir "${directory}/src")
    file(MAKE_DIRECTORY "${source_dir}")

    set(params_file "${source_dir}/_gen_params.txt")
    set(current_params "${ARGN}")
    set(params "")
    if(EXISTS "${params_file}")
        file(READ "${params_file}" params)
    endif()
    if(NOT params OR NOT(params STREQUAL current_params))
        message("Downloading external: ${namespace}")
        file(REMOVE_RECURSE "${source_dir}")
        include(FetchContent)
        FetchContent_Declare(
            ${namespace}_external_content
            ${ARGN}
        )
        FetchContent_GetProperties(${namespace}_external_content)
        if(NOT ${namespace}_external_content_POPULATED)
            FetchContent_Populate(${namespace}_external_content)
            prep_thirdparty(${${namespace}_external_content_SOURCE_DIR} "${source_dir}")
        endif()
        file(WRITE "${params_file}" "${current_params}")
    endif()
endfunction()
