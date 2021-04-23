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
    set(oneValueArgs INSTALL CUSTOM_PRE_CMAKE USE_PYTHON3 SOURCE_COMMAND)
    set(multiValueArgs TARGETS PROJECT_OPTIONS EXTERNAL_CXX_FLAGS EXTERNAL_C_FLAGS DEPENDENCIES)
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
        USE_PYTHON3 ${EXTERNAL_USE_PYTHON3}
        SOURCE_COMMAND ${EXTERNAL_SOURCE_COMMAND}
        INSTALL ${EXTERNAL_INSTALL}
        CUSTOM_PRE_CMAKE "${EXTERNAL_CUSTOM_PRE_CMAKE}"
        TARGETS ${EXTERNAL_TARGETS}
        DEPENDENCIES ${EXTERNAL_DEPENDENCIES}
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
endfunction()
