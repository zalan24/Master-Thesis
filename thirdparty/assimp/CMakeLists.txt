# include(FetchContent)

# FetchContent_Declare(
#     assimp_external
#     GIT_REPOSITORY https://github.com/assimp/assimp.git
#     GIT_TAG        v5.0.1
# )

# if ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" )
#   # Workaround for assimp's terrible build system
#   add_compile_options(-Wno-unused-command-line-argument -Wno-c99-extensions)
# endif()

# FetchContent_GetProperties(assimp_external)
# if(NOT assimp_external_POPULATED)
#   FetchContent_Populate(assimp_external)
#   set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
#   prep_thirdparty(${assimp_external_SOURCE_DIR} "${assimp_external_SOURCE_DIR}_fixed")
#   add_subdirectory("${assimp_external_SOURCE_DIR}_fixed" ${assimp_external_BINARY_DIR})
# endif()
# set(FIXED "${assimp_external_SOURCE_DIR}_fixed")

# include(Download)
# include(AddExternal)

# download(
#     # destination directory, source code will be in subdirectory `src`
#     ${PROJECT_BINARY_DIR}/3rdParty/assimp
#     # Standard CMAKE ExternalProject parameters starting from here
#     GIT_REPOSITORY https://github.com/assimp/assimp.git
#     GIT_TAG v5.0.1
# )

# add_external(
#     # namespace for the targets
#     assimp
#     # source directory
#     ${PROJECT_BINARY_DIR}/3rdParty/assimp/src
#     # settings for the build
#     ASSIMP_BUILD_TESTS=OFF
#     # -Wno-unused-command-line-argument
#     # -Wno-c99-extensions
# )


include(ExternalProject)
ExternalProject_Add(assimp_external
	GIT_REPOSITORY https://github.com/assimp/assimp.git
	GIT_TAG "v5.0.1"
	SOURCE_DIR "${PROJECT_BINARY_DIR}/3rdParty/assimp-src"
	BINARY_DIR "${PROJECT_BINARY_DIR}/3rdParty/assimp-build"
	# UPDATE_COMMAND ""
  UPDATE_DISCONNECTED ON
	# CONFIGURE_COMMAND ""
	# BUILD_COMMAND ""
	# INSTALL_COMMAND ""
	# TEST_COMMAND ""
  LOG_DOWNLOAD ON
  CMAKE_CACHE_ARGS
    -DCMAKE_CXX_COMPILER:PATH=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_COMPILER_AR:PATH=${CMAKE_CXX_COMPILER_AR}
    -DCMAKE_CXX_COMPILER_RANLIB:PATH=${CMAKE_CXX_COMPILER_RANLIB}
    -DCMAKE_C_COMPILER:PATH=${CMAKE_C_COMPILER}
    -DCMAKE_C_COMPILER_AR:PATH=${CMAKE_C_COMPILER_AR}
    -DCMAKE_C_COMPILER_RANLIB:PATH=${CMAKE_C_COMPILER_RANLIB}
    -DCMAKE_LINKER:PATH=${CMAKE_LINKER}
  CMAKE_ARGS
    -Wno-dev
    -DASSIMP_BUILD_TESTS=OFF
    -DCMAKE_INSTALL_PREFIX:PATH=${PROJECT_BINARY_DIR}/external/assimp
)

add_library(ASSIMP_LIB INTERFACE)
add_dependencies(ASSIMP_LIB assimp_external)
# target_include_directories(ASSIMP_LIB SYSTEM INTERFACE "${assimp_external_BINARY_DIR}/include" "${FIXED}/include")
# target_link_libraries(ASSIMP_LIB INTERFACE assimp)
# target_include_directories(ASSIMP_LIB SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/assimp/src/include)
# target_link_libraries(ASSIMP_LIB INTERFACE assimp)

# include("${PROJECT_BINARY_DIR}/external/assimp/lib/cmake/assimp-5.0/assimp-config-version.cmake")
# include("${PROJECT_BINARY_DIR}/external/assimp/lib/cmake/assimp-5.0/assimp-config.cmake")

target_include_directories(ASSIMP_LIB SYSTEM INTERFACE "${PROJECT_BINARY_DIR}/external/assimp/include")
target_link_directories(ASSIMP_LIB INTERFACE "${PROJECT_BINARY_DIR}/external/assimp/lib")
# target_link_libraries(ASSIMP_LIB INTERFACE assimp::assimp)
