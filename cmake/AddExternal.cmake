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
    set(oneValueArgs INSTALL)
    set(multiValueArgs TARGETS PROJECT_OPTIONS)
    cmake_parse_arguments(ADD_EXTERNAL "${options}" "${oneValueArgs}"
                                        "${multiValueArgs}" ${ARGN} )

    # message("${namespace} ------ ${ADD_EXTERNAL_TARGETS}")
    # message("${namespace} ------ ${ADD_EXTERNAL_PROJECT_OPTIONS}")

    # set(collect_snippet [=[
    # cmake_policy(SET CMP0026 OLD)
    # set(_all_targets "" CACHE INTERNAL "")
    # macro(add_library name)
    #     set(_all_targets "${_all_targets};${name}" CACHE INTERNAL "")
    #     _add_library(${name} ${ARGN})
    # endmacro()

    # macro(add_executable name)
    #     set(_all_targets "${_all_targets};${name}" CACHE INTERNAL "")
    #     _add_executable(${name} ${ARGN})
    # endmacro()
    # ]=])

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
    # add_custom_target(_@namespace@_external_gen ALL
    #                   COMMAND ${CMAKE_COMMAND} -E echo "${_out_str}" > "${_targets_file}"
    #                   DEPENDS @ADD_EXTERNAL_TARGETS@
    #                   BYPRODUCTS ${_target_file}
    #                   COMMENT "Creating cmake targets file for @namespace@: ${target_file}"
    # )

    # set(_out_str "")
    # foreach(target @ADD_EXTERNAL_TARGETS@)
    #     set(location "$<TARGET_FILE:${target}>")
    #     set(_out_str "${_out_str}\nadd_library(${target} STATIC IMPORTED)")
    #     set(_out_str "${_out_str}\nset_property(TARGET ${target} PROPERTY IMPORTED_LOCATION \"${location}\")")
    # endforeach()
    # set(_out_str "${_out_str}\nadd_library(${namespace}_external INTERFACE)")
    # set(_out_str "${_out_str}\ntarget_link_libraries(_@namespace@_external INTERFACE @ADD_EXTERNAL_TARGETS@)")

    # message("aoeuaoeuaoue ------- ${_out_str}")
    # # add_custom_target(_@namespace@_external_gen ALL
    # #     )
    # add_library(${namespace}_external INTERFACE)
    # target_link_libraries(_@namespace@_external INTERFACE @ADD_EXTERNAL_TARGETS@)
    # # add_dependencies(_@namespace@_external _@namespace@_external_gen)


    # file(WRITE "${_targets_file}" "")
    # foreach(target ${_all_targets})
    #     if (TARGET ${target})
    #         get_target_property(imported ${target} IMPORTED)
    #         get_target_property(aliased ${target} ALIASED_TARGET)
    #         get_target_property(target_type ${target} TYPE)
    #         if ("${target_type}" MATCHES "^(STATIC_LIBRARY|SHARED_LIBRARY|EXECUTABLE)$")
    #             if(imported)
    #                 get_target_property(location ${target} LOCATION)
    #                 if(location)
    #                     file(APPEND "${_targets_file}" "add_library(${target} STATIC IMPORTED)\nset_property(TARGET ${target} PROPERTY IMPORTED_LOCATION \"${location}\")\n")
    #                 endif()
    #             elseif(aliased)
    #                 get_target_property(location ${target} LOCATION)
    #                 if(location)
    #                     file(APPEND "${_targets_file}" "add_library(@namespace@::${target} STATIC IMPORTED)\nset_property(TARGET @namespace@::${target} PROPERTY IMPORTED_LOCATION \"${location}\")\n")
    #                 endif()
    #             else()
    #                 list(APPEND export_targets ${target})
    #             endif()
    #         endif()
    #     endif()
    # endforeach()
    # file(APPEND "${_targets_file}" "set(_@namespace@_targets ${export_targets})\n\n")
    # export(TARGETS ${export_targets} NAMESPACE @namespace@:: APPEND FILE "${_targets_file}")

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

    # patch CMakeLists.txt if not patched or updated
    if(EXISTS "${source_dir}/CMakeLists.txt.hash")
        file(READ "${source_dir}/CMakeLists.txt.hash" should_hash)
        file(SHA1 "${source_dir}/CMakeLists.txt" is_hash)
    endif()
    if(NOT is_hash OR NOT(is_hash STREQUAL should_hash))
        file(READ "${source_dir}/CMakeLists.txt" cmake_list)
        file(WRITE "${source_dir}/CMakeLists.txt" "${cmake_list}\n\ninclude(_target_export.cmake)\n")
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

    if (NOT EXISTS "${_targets_file}")
        message("Building external: ${namespace}")
        string(CONFIGURE "${export_snippet}" target_export @ONLY)
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
    endif()

    if (${ADD_EXTERNAL_INSTALL})
        add_library(${namespace}_external INTERFACE)
        target_include_directories(${namespace}_external SYSTEM INTERFACE "${PROJECT_BINARY_DIR}/3rdParty/${namespace}/install/include")
        foreach(target ${ADD_EXTERNAL_TARGETS})
            include("${PROJECT_BINARY_DIR}/3rdParty/${namespace}/install/lib/cmake/${target}Targets.cmake")
        endforeach()
        target_link_libraries(${namespace}_external INTERFACE ${ADD_EXTERNAL_TARGETS})
    else()
        include("${_targets_file}")
    endif()

    # # foreach(target IN LISTS _${namespace}_targets)
    # #     if(TARGET ${namespace}::${target})
    # #         get_target_property(location ${namespace}::${target} LOCATION)
    # #         add_custom_command(
    # #             OUTPUT "${location}"
    # #             COMMAND cmake --build . --config ${CMAKE_BUILD_TYPE} --target ${target}
    # #             WORKING_DIRECTORY "${binary_dir}"
    # #             VERBATIM USES_TERMINAL
    # #             COMMENT "Generating ${namespace}::${target}"
    # #         )
    # #         add_custom_target(_${namespace}_${target}_compile
    # #             DEPENDS ${location}
    # #         )
    # #         add_dependencies(${namespace}::${target} _${namespace}_${target}_compile)
    # #     endif()
    # # endforeach()
endfunction()
