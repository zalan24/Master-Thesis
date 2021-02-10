function(download_external namespace)
    include(download)
    download(
        ${PROJECT_BINARY_DIR}/3rdParty/${namespace}
        ${ARGN}
    )
endfunction()

function(external namespace)

    include(AddExternal)

    set(options)
    set(oneValueArgs INSTALL)
    set(multiValueArgs TARGETS PROJECT_OPTIONS EXTERNAL_CXX_FLAGS EXTERNAL_C_FLAGS)
    cmake_parse_arguments(EXTERNAL "${options}" "${oneValueArgs}"
                                    "${multiValueArgs}" ${ARGN} )


    list(JOIN EXTERNAL_EXTERNAL_CXX_FLAGS " " external_cxx_flags)
    list(JOIN EXTERNAL_EXTERNAL_C_FLAGS " " external_c_flags)

    # F you, cmake
    add_external(
        # namespace for the targets
        ${namespace}
        # source directory
        ${PROJECT_BINARY_DIR}/3rdParty/${namespace}/src
        INSTALL ${EXTERNAL_INSTALL}
        TARGETS ${EXTERNAL_TARGETS}
        PROJECT_OPTIONS
            ${EXTERNAL_PROJECT_OPTIONS}
            -DCMAKE_INSTALL_PREFIX:PATH=${PROJECT_BINARY_DIR}/3rdParty/${namespace}/install
            -Wno-dev
            "-DCMAKE_CXX_COMPILER:PATH=${CMAKE_CXX_COMPILER}"
            "-DCMAKE_CXX_COMPILER_AR:PATH=${CMAKE_CXX_COMPILER_AR}"
            "-DCMAKE_CXX_COMPILER_RANLIB:PATH=${CMAKE_CXX_COMPILER_RANLIB}"
            "-DCMAKE_C_COMPILER:PATH=${CMAKE_C_COMPILER}"
            "-DCMAKE_C_COMPILER_AR:PATH=${CMAKE_C_COMPILER_AR}"
            "-DCMAKE_C_COMPILER_RANLIB:PATH=${CMAKE_C_COMPILER_RANLIB}"
            "-DCMAKE_LINKER:PATH=${CMAKE_LINKER}"
            "-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS} ${external_cxx_flags}"
            "-DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}"
            "-DCMAKE_CXX_FLAGS_MINSIZEREL:STRING=${CMAKE_CXX_FLAGS_MINSIZEREL}"
            "-DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}"
            "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}"
            "-DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS} ${external_c_flags}"
            "-DCMAKE_C_FLAGS_DEBUG:STRING=${CMAKE_C_FLAGS_DEBUG}"
            "-DCMAKE_C_FLAGS_MINSIZEREL:STRING=${CMAKE_C_FLAGS_MINSIZEREL}"
            "-DCMAKE_C_FLAGS_RELEASE:STRING=${CMAKE_C_FLAGS_RELEASE}"
            "-DCMAKE_C_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_C_FLAGS_RELWITHDEBINFO}"
            "-DCMAKE_EXE_LINKER_FLAGS:STRING=${CMAKE_EXE_LINKER_FLAGS}"
            "-DCMAKE_EXE_LINKER_FLAGS_DEBUG:STRING=${CMAKE_EXE_LINKER_FLAGS_DEBUG}"
            "-DCMAKE_EXE_LINKER_FLAGS_MINSIZEREL:STRING=${CMAKE_EXE_LINKER_FLAGS_MINSIZEREL}"
            "-DCMAKE_EXE_LINKER_FLAGS_RELEASE:STRING=${CMAKE_EXE_LINKER_FLAGS_RELEASE}"
            "-DCMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO}"
            "-DCMAKE_MODULE_LINKER_FLAGS:STRING=${CMAKE_MODULE_LINKER_FLAGS}"
            "-DCMAKE_MODULE_LINKER_FLAGS_DEBUG:STRING=${CMAKE_MODULE_LINKER_FLAGS_DEBUG}"
            "-DCMAKE_MODULE_LINKER_FLAGS_MINSIZEREL:STRING=${CMAKE_MODULE_LINKER_FLAGS_MINSIZEREL}"
            "-DCMAKE_MODULE_LINKER_FLAGS_RELEASE:STRING=${CMAKE_MODULE_LINKER_FLAGS_RELEASE}"
            "-DCMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO}"
            "-DCMAKE_RC_FLAGS:STRING=${CMAKE_RC_FLAGS}"
            "-DCMAKE_RC_FLAGS_DEBUG:STRING=${CMAKE_RC_FLAGS_DEBUG}"
            "-DCMAKE_RC_FLAGS_MINSIZEREL:STRING=${CMAKE_RC_FLAGS_MINSIZEREL}"
            "-DCMAKE_RC_FLAGS_RELEASE:STRING=${CMAKE_RC_FLAGS_RELEASE}"
            "-DCMAKE_RC_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_RC_FLAGS_RELWITHDEBINFO}"
            "-DCMAKE_SHARED_LINKER_FLAGS:STRING=${CMAKE_SHARED_LINKER_FLAGS}"
            "-DCMAKE_SHARED_LINKER_FLAGS_DEBUG:STRING=${CMAKE_SHARED_LINKER_FLAGS_DEBUG}"
            "-DCMAKE_SHARED_LINKER_FLAGS_MINSIZEREL:STRING=${CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL}"
            "-DCMAKE_SHARED_LINKER_FLAGS_RELEASE:STRING=${CMAKE_SHARED_LINKER_FLAGS_RELEASE}"
            "-DCMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO}"
            "-DCMAKE_STATIC_LINKER_FLAGS:STRING=${CMAKE_STATIC_LINKER_FLAGS}"
            "-DCMAKE_STATIC_LINKER_FLAGS_DEBUG:STRING=${CMAKE_STATIC_LINKER_FLAGS_DEBUG}"
            "-DCMAKE_STATIC_LINKER_FLAGS_MINSIZEREL:STRING=${CMAKE_STATIC_LINKER_FLAGS_MINSIZEREL}"
            "-DCMAKE_STATIC_LINKER_FLAGS_RELEASE:STRING=${CMAKE_STATIC_LINKER_FLAGS_RELEASE}"
            "-DCMAKE_STATIC_LINKER_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_STATIC_LINKER_FLAGS_RELWITHDEBINFO}"
    )

    # add_library(LIB_EXTERNAL_${namespace} INTERFACE)
    # target_include_directories(LIB_EXTERNAL_${namespace} SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/${namespace}/src)
    # target_link_libraries(LIB_EXTERNAL_${namespace} INTERFACE glslang::glslang glslang::SPIRV)

    # set(directory "${PROJECT_BINARY_DIR}/3rdParty")
    # set(external_dir "${PROJECT_BINARY_DIR}/external")

    # set(download_dir "${directory}/${namespace}/download")
    # set(source_dir "${directory}/${namespace}/src")
    # set(binary_dir "${directory}/${namespace}/build")
    # set(install_dir "${external_dir}/${namespace}")

    # set(external_name "${namespace}_external")

    # include(ExternalProject)
    # ExternalProject_Add(${external_name}
    #     SOURCE_DIR "${source_dir}"
    #     BINARY_DIR "${binary_dir}"
    #     DOWNLOAD_DIR "${download_dir}"
    #     # UPDATE_COMMAND ""
    #     UPDATE_DISCONNECTED ON
    #     # CONFIGURE_COMMAND ""
    #     # BUILD_COMMAND ""
    #     # INSTALL_COMMAND ""
    #     # TEST_COMMAND ""
    #     ${ARGN}
    #     LOG_DOWNLOAD ON
    #     CMAKE_CACHE_ARGS
    #         -DCMAKE_CXX_COMPILER:PATH=${CMAKE_CXX_COMPILER}
    #         -DCMAKE_CXX_COMPILER_AR:PATH=${CMAKE_CXX_COMPILER_AR}
    #         -DCMAKE_CXX_COMPILER_RANLIB:PATH=${CMAKE_CXX_COMPILER_RANLIB}
    #         -DCMAKE_C_COMPILER:PATH=${CMAKE_C_COMPILER}
    #         -DCMAKE_C_COMPILER_AR:PATH=${CMAKE_C_COMPILER_AR}
    #         -DCMAKE_C_COMPILER_RANLIB:PATH=${CMAKE_C_COMPILER_RANLIB}
    #         -DCMAKE_LINKER:PATH=${CMAKE_LINKER}
    #     CMAKE_ARGS
    #         -Wno-dev
    #         -DCMAKE_INSTALL_PREFIX:PATH=${install_dir}
    # )

    # string(CONFIGURE "${download_template}" cmake_lists @ONLY)
    # file(WRITE "${download_dir}/CMakeLists.txt" "${cmake_lists}")
    # execute_process(COMMAND ${CMAKE_COMMAND} .
    #                 RESULT_VARIABLE result
    #                 WORKING_DIRECTORY "${download_dir}"
    # )
    # if(result)
    #     message(FATAL_ERROR "CMake step for ${name} failed: ${result}")
    # endif()
    # execute_process(COMMAND ${CMAKE_COMMAND} --build .
    #                 RESULT_VARIABLE result
    #                 WORKING_DIRECTORY "${download_dir}"
    # )
    # if(result)
    #     message(FATAL_ERROR "Build step for ${name} failed: ${result}")
    # endif()
endfunction()