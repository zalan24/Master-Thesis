include(external)

download_external(
    vulkan_tools
    GIT_REPOSITORY https://github.com/LunarG/VulkanTools.git
	GIT_TAG        v1.2.176
)

external(
    vulkan_tools
    INSTALL ON
    USE_PYTHON3 ON
    TARGETS
    PROJECT_OPTIONS
        -DVULKAN_HEADERS_INSTALL_DIR=${PROJECT_BINARY_DIR}/3rdParty/vulkan_headers/install
        -DVULKAN_VALIDATIONLAYERS_INSTALL_DIR=${PROJECT_BINARY_DIR}/3rdParty/validation_layers/install
        -DVULKAN_LOADER_INSTALL_DIR=${PROJECT_BINARY_DIR}/3rdParty/vulkan_loader/install
)

target_link_libraries(vulkan_tools_external INTERFACE vulkan_headers_external vulkan_loader_external vulkan_headers_external)
