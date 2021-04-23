include(external)

download_external(
    vulkan_extensions
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-ExtensionLayer.git
	GIT_TAG        sdk-1.2.170.0
)

external(
    vulkan_extensions
    INSTALL ON
    USE_PYTHON3 ON
    TARGETS
    PROJECT_OPTIONS
        -DVULKAN_HEADERS_INSTALL_DIR=${PROJECT_BINARY_DIR}/3rdParty/vulkan_headers/install
        -DGLSLANG_INSTALL_DIR=${PROJECT_BINARY_DIR}/3rdParty/glslang/install
        -DSPIRV_HEADERS_INSTALL_DIR=${PROJECT_BINARY_DIR}/3rdParty/spirv_headers/install
        # -DVULKAN_VALIDATIONLAYERS_INSTALL_DIR=${PROJECT_BINARY_DIR}/3rdParty/validation_layers/install
        # -DVULKAN_LOADER_INSTALL_DIR=${PROJECT_BINARY_DIR}/3rdParty/vulkan_loader/install
)

target_link_libraries(vulkan_extensions_external INTERFACE
    vulkan_headers_external
    glslang_external
    spirv_headers_external
)
