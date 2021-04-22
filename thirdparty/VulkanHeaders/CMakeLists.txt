include(external)

download_external(
    vulkan_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG        v1.2.176
)

external(
    vulkan_headers
    INSTALL ON
)
