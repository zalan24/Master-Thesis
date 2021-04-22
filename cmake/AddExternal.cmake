###########################################################################
#   Copyright 2017 Florian Reiterer
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
###########################################################################

function(add_external namespace source_dir)

    set(options)
    set(oneValueArgs INSTALL CUSTOM_PRE_CMAKE)
    set(multiValueArgs TARGETS PROJECT_OPTIONS DEPENDENCIES)
    cmake_parse_arguments(ADD_EXTERNAL "${options}" "${oneValueArgs}"
                                        "${multiValueArgs}" ${ARGN} )

    set(export_snippet [=[
        set(_out_str "")
        foreach(target @ADD_EXTERNAL_TARGETS@)
            set(location "$<TARGET_FILE:${target}>")
            set(_out_str "${_out_str}\nadd_library(${target} STATIC IMPORTED GLOBAL)")
            set(_out_str "${_out_str}\nset_property(TARGET ${target} PROPERTY IMPORTED_LOCATION \"${location}\")")
        endforeach()
        set(_out_str "${_out_str}\nadd_library(@namespace@_external INTERFACE)")
        set(_out_str "${_out_str}\ntarget_link_libraries(@namespace@_external INTERFACE @ADD_EXTERNAL_TARGETS@)")
        set(_out_str "${_out_str}\nif (EXISTS \"@source_dir@/include\")")
        set(_out_str "${_out_str}\n    target_include_directories(@namespace@_external SYSTEM INTERFACE \"@source_dir@/include\")")
        set(_out_str "${_out_str}\nelse()")
        set(_out_str "${_out_str}\n    target_include_directories(@namespace@_external SYSTEM INTERFACE \"@source_dir@\")")
        set(_out_str "${_out_str}\nendif()")
        set(_out_str "${_out_str}\nif (EXISTS \"@binary_dir@/include\")")
        set(_out_str "${_out_str}\n    target_include_directories(@namespace@_external SYSTEM INTERFACE \"@binary_dir@/include\")")
        set(_out_str "${_out_str}\nelse()")
        set(_out_str "${_out_str}\n    target_include_directories(@namespace@_external SYSTEM INTERFACE \"@binary_dir@\")")
        set(_out_str "${_out_str}\nendif()")

        file(GENERATE OUTPUT "@_targets_file@" CONTENT "${_out_str}")
        add_library(_@namespace@_external INTERFACE)
        target_link_libraries(_@namespace@_external INTERFACE @ADD_EXTERNAL_TARGETS@)
        # add_dependencies(_@namespace@_external _@namespace@_external_gen)
    ]=])

    get_filename_component(source_dir "${source_dir}" ABSOLUTE)
    if(NOT EXISTS "${source_dir}")
        message(FATAL_ERROR "Source directory \"${source_dir}\" not found")
    endif()
    set(binary_dir "${CMAKE_BINARY_DIR}/3rdParty/${namespace}/bin")
    set(cmake_dir "${CMAKE_BINARY_DIR}/external")
    # set(install_dir "${CMAKE_BINARY_DIR}/3rdParty/${namespace}/install")
    file(MAKE_DIRECTORY "${binary_dir}")
    file(MAKE_DIRECTORY "${cmake_dir}")
    # file(MAKE_DIRECTORY "${install_dir}")


    set(_targets_file "${cmake_dir}/${namespace}Targets.cmake")
    set(params_file "${source_dir}/_params.txt")

    # patch CMakeLists.txt if not patched or updated
    if(EXISTS "${source_dir}/CMakeLists.txt.hash")
        file(READ "${source_dir}/CMakeLists.txt.hash" should_hash)
        file(SHA1 "${source_dir}/CMakeLists.txt" is_hash)
    endif()
    if(NOT is_hash OR NOT(is_hash STREQUAL should_hash))
        file(READ "${source_dir}/CMakeLists.txt" cmake_list)
        if (NOT ("${ADD_EXTERNAL_CUSTOM_PRE_CMAKE}" STREQUAL ""))
            file(READ "${ADD_EXTERNAL_CUSTOM_PRE_CMAKE}" PRE_CMAKE)
        endif()
        foreach(dep ${ADD_EXTERNAL_DEPENDENCIES})
            set(PRE_CMAKE "${PRE_CMAKE}\ninclude(\"${dep}\")")
            # string(REPLACE "/install/lib/cmake/" "/" target_path "${dep}")
            # foreach(target ${ADD_EXTERNAL_TARGETS})
            #     string(REPLACE "::" "/" target_path "${target}")
            #     include("${PROJECT_BINARY_DIR}/3rdParty/${namespace}/install/lib/cmake/${target_path}Targets.cmake")
            # endforeach()
        endforeach()
        file(WRITE "${source_dir}/CMakeLists.txt" "${PRE_CMAKE}\n\n${cmake_list}\n\ninclude(_target_export.cmake)\n")
        file(SHA1 "${source_dir}/CMakeLists.txt" hash)
        file(WRITE "${source_dir}/CMakeLists.txt.hash" "${hash}")
    endif()

    foreach(param IN LISTS ADD_EXTERNAL_PROJECT_OPTIONS)
        if(param MATCHES "^[A-Za-z_]+=.+$")
            list(APPEND cmake_params "-D${param}")
        elseif(param MATCHES "^-.+$")
            list(APPEND cmake_params "${param}")
        else()
            message(FATAL_ERROR "Invalid parameter \"${param}\"")
        endif()
    endforeach()

    file(TIMESTAMP "${source_dir}/CMakeLists.txt" cmake_time_stamp)
    string(CONFIGURE "@ADD_EXTERNAL_TARGETS@ ;; @ADD_EXTERNAL_PROJECT_OPTIONS@ ;; @ADD_EXTERNAL_INSTALL@ ;; @CMAKE_GENERATOR_PLATFORM@ ;; @CMAKE_GENERATOR@ ;; @CMAKE_BUILD_TYPE@ ;; @cmake_params@ ;; @cmake_time_stamp@" build_options @ONLY)

    if(EXISTS "${params_file}")
        file(READ "${params_file}" old_build_options)
    endif()

    if (NOT old_build_options OR NOT(old_build_options STREQUAL build_options))
        message("Building external: ${namespace}")
        if (${ADD_EXTERNAL_INSTALL})
            set(target_export "")
        else()
            string(CONFIGURE "${export_snippet}" target_export @ONLY)
        endif()
        file(WRITE "${source_dir}/_target_export.cmake" "${target_export}")
        execute_process(COMMAND "${CMAKE_COMMAND}" -G${CMAKE_GENERATOR} -DCMAKE_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -Wno-dev ${cmake_params} "${source_dir}"
                        RESULT_VARIABLE result
                        WORKING_DIRECTORY "${binary_dir}"
        )
        if(result)
            message(FATAL_ERROR "CMake step for ${namespace} failed: ${result}")
        endif()
        execute_process(COMMAND "${CMAKE_COMMAND}" --build . --config ${CMAKE_BUILD_TYPE} # --target _${namespace}_external
                        RESULT_VARIABLE result
                        WORKING_DIRECTORY "${binary_dir}"
        )
        if(result)
            message(FATAL_ERROR "CMake build for ${namespace} failed: ${result}")
        endif()
        if (${ADD_EXTERNAL_INSTALL})
            execute_process(COMMAND "${CMAKE_COMMAND}" --install . --config ${CMAKE_BUILD_TYPE} # --target _${namespace}_external
                            RESULT_VARIABLE result
                            WORKING_DIRECTORY "${binary_dir}"
            )
            if(result)
                message(FATAL_ERROR "CMake install for ${namespace} failed: ${result}")
            endif()
        endif()
        file(WRITE "${params_file}" "${build_options}")
    endif()

    if (${ADD_EXTERNAL_INSTALL})
        add_library(${namespace}_external INTERFACE)
        target_include_directories(${namespace}_external SYSTEM INTERFACE "${PROJECT_BINARY_DIR}/3rdParty/${namespace}/install/include")
        foreach(target ${ADD_EXTERNAL_TARGETS})
            string(REPLACE "::" "/" target_path "${target}")
            include("${PROJECT_BINARY_DIR}/3rdParty/${namespace}/install/lib/cmake/${target_path}Targets.cmake")
        endforeach()
        target_link_libraries(${namespace}_external INTERFACE ${ADD_EXTERNAL_TARGETS})
    else()
        include("${_targets_file}")
    endif()
endfunction()
