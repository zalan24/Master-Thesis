include(external)

download_external(
    vulkan_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
)

external(
    vulkan_headers
    INSTALL ON
)
