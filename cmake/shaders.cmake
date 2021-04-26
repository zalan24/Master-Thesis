function(preprocess_shaders)
    set(options)
    set(oneValueArgs TARGET OUT_DIR)
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

    # set(ClearTarget "Clear_${SHADERS_TARGET}")
    # add_custom_target(
    #     ${ClearTarget}
    #     COMMAND  ${CMAKE_COMMAND} -E remove "${REGISTRY_FILE}"
    #     DEPENDS  ShaderPreprocessor
    #     COMMENT "Removing shader cache: ${REGISTRY_FILE}"
    # )

    add_custom_command(
        OUTPUT   ${GENERATED_FILES}
        COMMAND  ShaderPreprocessor
            --output ${SHADERS_OUT_DIR}
            --target "${SHADERS_OUT_DIR}/${SHADERS_TARGET}.json"
            --headers ${SHADERS_HEADERS}
            --sources ${SHADERS_SOURCES}
        DEPENDS  ${SHADERS_HEADERS} ${SHADERS_SOURCES} ShaderPreprocessor
    )


    add_library(${SHADERS_TARGET} STATIC ${GENERATED_FILES})
    # add_dependencies(${SHADERS_TARGET} ShaderCodes)

    target_include_directories(${SHADERS_TARGET} PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${SHADERS_OUT_DIR})
    target_link_libraries(${SHADERS_TARGET} PUBLIC ShaderManager)
endfunction()