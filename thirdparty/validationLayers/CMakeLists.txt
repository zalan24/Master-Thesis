include(external)

download_external(
    validation_layers
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-ValidationLayers.git
    GIT_TAG        v1.2.170
)

external(
    validation_layers
    INSTALL ON
    PROJECT_OPTIONS
        -DBUILD_TESTS=OFF
        -DVULKAN_HEADERS_INSTALL_DIR=${PROJECT_BINARY_DIR}/3rdParty/vulkan_headers/install
        -DGLSLANG_INSTALL_DIR=${PROJECT_BINARY_DIR}/3rdParty/glslang/install
        -DSPIRV_HEADERS_INSTALL_DIR=${PROJECT_BINARY_DIR}/3rdParty/spirv_headers/install
        -DSPIRV_TOOLS_INSTALL_DIR=${PROJECT_BINARY_DIR}/3rdParty/spirv_tools/install
        -DROBIN_HOOD_HASHING_INSTALL_DIR=${PROJECT_BINARY_DIR}/3rdParty/robin_hood_hash/install
)

target_link_libraries(validation_layers_external INTERFACE
    vulkan_headers_external
    glslang_external
    spirv_headers_external
    spirv_tools_external
    robin_hood_hash_external
)
