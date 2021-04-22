include(external)

download_external(
    validation_layers
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-ValidationLayers.git
    GIT_TAG        v1.2.176
)

set(external_cxx_flags "")
set(external_c_flags "")
if ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" )
  # Workaround for assimp's terrible build system
  set(external_cxx_flags -Wno-unused-command-line-argument -Wno-c99-extensions)
  set(external_c_flags -Wno-unused-command-line-argument)
endif()

external(
    validation_layers
    INSTALL ON
    DEPENDENCIES
        "${PROJECT_BINARY_DIR}/3rdParty/spirv_tools/install/SPIRV-Tools/cmake/SPIRV-ToolsTarget.cmake"
        "${PROJECT_BINARY_DIR}/3rdParty/spirv_tools/install/SPIRV-Tools-opt/cmake/SPIRV-Tools-optTargets.cmake"
        "${PROJECT_BINARY_DIR}/3rdParty/robin_hood_hash/install/lib/cmake/robin_hood/robin_hood-targets.cmake"
    PROJECT_OPTIONS
        -DSPIRV_WERROR=OFF
        -DBUILD_TESTS=OFF
        -DBUILD_WERROR=OFF
        -DUSE_ROBIN_HOOD_HASHING=OFF
        -DVULKAN_HEADERS_INSTALL_DIR=${PROJECT_BINARY_DIR}/3rdParty/vulkan_headers/install
        -DGLSLANG_INSTALL_DIR=${PROJECT_BINARY_DIR}/3rdParty/glslang/install
        -DSPIRV_HEADERS_INSTALL_DIR=${PROJECT_BINARY_DIR}/3rdParty/spirv_headers/install
        -DSPIRV_TOOLS_INSTALL_DIR=${PROJECT_BINARY_DIR}/3rdParty/spirv_tools/install
        -DROBIN_HOOD_HASHING_INSTALL_DIR=${PROJECT_BINARY_DIR}/3rdParty/robin_hood_hash/install
    EXTERNAL_CXX_FLAGS ${external_cxx_flags}
    EXTERNAL_C_FLAGS ${external_c_flags}
)

target_link_libraries(validation_layers_external INTERFACE
    vulkan_headers_external
    glslang_external
    spirv_headers_external
    spirv_tools_external
    robin_hood_hash_external
)
