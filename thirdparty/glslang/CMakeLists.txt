# include(FetchContent)

# FetchContent_Declare(
#     glslang_external
#     GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
#     GIT_TAG        11.1.0
# )

# FetchContent_GetProperties(glslang_external)
# if(NOT glslang_external_POPULATED)
#   FetchContent_Populate(glslang_external)
#   prep_thirdparty(${glslang_external_SOURCE_DIR} "${glslang_external_SOURCE_DIR}_fixed")
#   add_subdirectory("${glslang_external_SOURCE_DIR}_fixed" ${glslang_external_BINARY_DIR})
# endif()
# set(FIXED "${glslang_external_SOURCE_DIR}_fixed")

# include(Download)
# include(AddExternal)

# message("--- download ---")

# download(
#     # destination directory, source code will be in subdirectory `src`
#     ${PROJECT_BINARY_DIR}/3rdParty/glslang
#     # Standard CMAKE ExternalProject parameters starting from here
#     GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
#     GIT_TAG 11.1.0
#     UPDATE_DISCONNECTED ON
#     CMAKE_CACHE_ARGS
#         -DCMAKE_CXX_COMPILER:PATH=${CMAKE_CXX_COMPILER}
#         -DCMAKE_CXX_COMPILER_AR:PATH=${CMAKE_CXX_COMPILER_AR}
#         -DCMAKE_CXX_COMPILER_RANLIB:PATH=${CMAKE_CXX_COMPILER_RANLIB}
#         -DCMAKE_C_COMPILER:PATH=${CMAKE_C_COMPILER}
#         -DCMAKE_C_COMPILER_AR:PATH=${CMAKE_C_COMPILER_AR}
#         -DCMAKE_C_COMPILER_RANLIB:PATH=${CMAKE_C_COMPILER_RANLIB}
#         -DCMAKE_LINKER:PATH=${CMAKE_LINKER}
# )

# message("--- add_external ---")

# add_external(
#     # namespace for the targets
#     glslang
#     # source directory
#     ${PROJECT_BINARY_DIR}/3rdParty/glslang/src
#     # settings for the build
#     BUILD_STATIC_LIBS=ON
#     -Wno-dev
# )

# message("--- done ---")

# set(SOURCE_DIR "${PROJECT_BINARY_DIR}/3rdParty/glslang-src")
# set(BINARY_DIR "${PROJECT_BINARY_DIR}/3rdParty/glslang-build")

# include(ExternalProject)
# ExternalProject_Add(glslang_external
# 	GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
# 	GIT_TAG "11.1.0"
# 	SOURCE_DIR "${SOURCE_DIR}"
# 	BINARY_DIR "${BINARY_DIR}"
# 	# UPDATE_COMMAND ""
#   UPDATE_DISCONNECTED ON
# 	# CONFIGURE_COMMAND ""
# 	# BUILD_COMMAND ""
# 	# INSTALL_COMMAND ""
# 	# TEST_COMMAND ""
#   LOG_DOWNLOAD ON
#   CMAKE_CACHE_ARGS
#     -DCMAKE_CXX_COMPILER:PATH=${CMAKE_CXX_COMPILER}
#     -DCMAKE_CXX_COMPILER_AR:PATH=${CMAKE_CXX_COMPILER_AR}
#     -DCMAKE_CXX_COMPILER_RANLIB:PATH=${CMAKE_CXX_COMPILER_RANLIB}
#     -DCMAKE_C_COMPILER:PATH=${CMAKE_C_COMPILER}
#     -DCMAKE_C_COMPILER_AR:PATH=${CMAKE_C_COMPILER_AR}
#     -DCMAKE_C_COMPILER_RANLIB:PATH=${CMAKE_C_COMPILER_RANLIB}
#     -DCMAKE_LINKER:PATH=${CMAKE_LINKER}
#   CMAKE_ARGS
#     -Wno-dev
#     -DCMAKE_INSTALL_PREFIX:PATH=${PROJECT_BINARY_DIR}/external/glslang
# )

# execute_process(COMMAND "${CMAKE_COMMAND}" -G${CMAKE_GENERATOR} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -Wno-dev ${cmake_params} "${SOURCE_DIR}"
#     RESULT_VARIABLE result
#     WORKING_DIRECTORY "${BINARY_DIR}"
# )

# if(result)
#     message(FATAL_ERROR "CMake step for glslang failed: ${result}")
# endif()


include(external)

download_external(
    glslang
    GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
	GIT_TAG        11.1.0
)

external(
    glslang
    INSTALL ON
    TARGETS
        OSDependent
        OGLCompiler
        glslang
        SPIRV
)

# add_library(Glslang_LIB INTERFACE)
# include("${CMAKE_BINARY_DIR}/external/glslangTargets.cmake")
# add_dependencies(Glslang_LIB glslang_external)

# find_package(glslang REQUIRED)

# include("${PROJECT_BINARY_DIR}/external/glslang/lib/cmake/OGLCompilerTargets.cmake")
# include("${PROJECT_BINARY_DIR}/external/glslang/lib/cmake/OSDependentTargets.cmake")
# include("${PROJECT_BINARY_DIR}/external/glslang/lib/cmake/glslangTargets.cmake")
# include("${PROJECT_BINARY_DIR}/external/glslang/lib/cmake/SPIRVTargets.cmake")

# target_include_directories(Glslang_LIB SYSTEM INTERFACE "${FIXED}")
# target_link_libraries(Glslang_LIB INTERFACE glslang SPIRV)
# target_include_directories(Glslang_LIB SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/glslang/src)
# target_link_libraries(Glslang_LIB INTERFACE glslang::glslang glslang::SPIRV)
# target_include_directories(Glslang_LIB SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/external/glslang/include)
# target_link_libraries(Glslang_LIB INTERFACE glslang SPIRV)
# target_link_libraries(Glslang_LIB INTERFACE glslang_external)
