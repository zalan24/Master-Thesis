include(external)

download_external(
    vulkan_hpp
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
)

add_library(vulkan_hpp_external INTERFACE)
target_include_directories(vulkan_hpp_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/vulkan_hpp/src/include)
