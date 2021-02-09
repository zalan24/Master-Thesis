include(external)

download_external(
    vulkan_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
)

add_library(vulkan_headers_external INTERFACE)
target_include_directories(vulkan_headers_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/vulkan_headers/src/include)
