include(external)

download_external(
    assimp
    GIT_REPOSITORY https://github.com/assimp/assimp.git
	  GIT_TAG        v5.0.1
)

set(external_cxx_flags "")
set(external_c_flags "")
if ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" )
  # Workaround for assimp's terrible build system
  set(external_cxx_flags -Wno-unused-command-line-argument -Wno-c99-extensions)
  set(external_c_flags -Wno-unused-command-line-argument)
endif()

external(
    assimp
    INSTALL OFF
    TARGETS
        assimp
    PROJECT_OPTIONS
        -DASSIMP_BUILD_TESTS=OFF
        -DBUILD_SHARED_LIBS=OFF
        -DASSIMP_NO_EXPORT=ON
        -DINJECT_DEBUG_POSTFIX=OFF
    EXTERNAL_CXX_FLAGS ${external_cxx_flags}
    EXTERNAL_C_FLAGS ${external_c_flags}
)
