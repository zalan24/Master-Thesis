include(external)

download_external(
    vulkan_loader
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Loader.git
	GIT_TAG        v1.2.176
)

external(
    vulkan_loader
    INSTALL ON
    TARGETS
    PROJECT_OPTIONS
        -DVULKAN_HEADERS_INSTALL_DIR=${PROJECT_BINARY_DIR}/3rdParty/vulkan_headers/install
)
