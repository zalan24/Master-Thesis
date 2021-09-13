function(preprocess_shaders)
    set(options)
    set(oneValueArgs TARGET OUT_DIR TARGETS_DIR DRIVER_REQUIREMENTS)
    set(multiValueArgs SOURCES HEADERS)
    cmake_parse_arguments(SHADERS "${options}" "${oneValueArgs}"
                                        "${multiValueArgs}" ${ARGN} )

    set(OUTPUT_DIR "${SHADERS_OUT_DIR}")
    set(REGISTRY_FILE "${OUTPUT_DIR}/shaderregistry.h")
    set(GENERATED_FILES "${REGISTRY_FILE}")

    foreach(H IN LISTS SHADERS_HEADERS SHADERS_SOURCES)
        get_filename_component(NAME ${H} NAME_WE)
        list(APPEND GENERATED_FILES "${OUTPUT_DIR}/shader_header_${NAME}.h" "${OUTPUT_DIR}/shader_header_${NAME}.cpp")
    endforeach()
    foreach(S IN LISTS SHADERS_SOURCES)
        get_filename_component(NAME ${S} NAME_WE)
        list(APPEND GENERATED_FILES "${OUTPUT_DIR}/shader_${NAME}.h" "${OUTPUT_DIR}/shader_${NAME}.cpp")
    endforeach()

    set(targetFile "${SHADERS_TARGETS_DIR}/${SHADERS_TARGET}.bin")

    add_custom_command(
        OUTPUT   ${GENERATED_FILES} ${targetFile}
        COMMAND  ShaderPreprocessor
            --output ${SHADERS_OUT_DIR}
            --target ${targetFile}
            --hardware ${SHADERS_DRIVER_REQUIREMENTS}
            --headers ${SHADERS_HEADERS}
            --sources ${SHADERS_SOURCES}
        DEPENDS  ${SHADERS_HEADERS} ${SHADERS_DRIVER_REQUIREMENTS} ${SHADERS_SOURCES} ShaderPreprocessor
    )


    add_library(${SHADERS_TARGET} STATIC ${GENERATED_FILES})

    target_include_directories(${SHADERS_TARGET} PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${SHADERS_OUT_DIR})
    target_link_libraries(${SHADERS_TARGET} PUBLIC ShaderManager)
endfunction()

function(compile_shaders)
    set(options)
    set(oneValueArgs TARGET_NAME DATA_DIR DRIVER_REQUIREMENTS SHADER_STATS COMPILE_OPTIONS OUTPUT CACHE_DIR)
    set(multiValueArgs TARGETS)
    cmake_parse_arguments(COMPILER "${options}" "${oneValueArgs}"
                                        "${multiValueArgs}" ${ARGN} )

    set(targetFiles)
    foreach(T IN ITEMS ${COMPILER_TARGETS})
        list(APPEND targetFiles "${COMPILER_DATA_DIR}/${T}.bin")
    endforeach()


    add_custom_command(
        OUTPUT   ${COMPILER_OUTPUT} # ${GENERATED_FILES}
        COMMAND  ShaderCompiler
            # -r${CMAKE_CURRENT_SOURCE_DIR}
            -c${COMPILER_CACHE_DIR}
            # --headers ${CMAKE_CURRENT_BINARY_DIR}
            -o${COMPILER_OUTPUT}
            # -d${SHADER_DEBUG}
            # -g${GENERATED_SHADERS}
            --hardware ${COMPILER_DRIVER_REQUIREMENTS}
            --options ${COMPILER_COMPILE_OPTIONS}
            # ${SHADERS_HEADERS} ${SHADERS_SOURCES}
            --files ${targetFiles}
        DEPENDS  ${targetFiles} ${COMPILER_DRIVER_REQUIREMENTS} ${COMPILER_COMPILE_OPTIONS} ShaderCompiler
    )

    add_custom_target(${COMPILER_TARGET_NAME} ALL DEPENDS ${COMPILER_OUTPUT})
endfunction()