include(FetchContent)

FetchContent_Declare(
    vulkan_header_external
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
)

FetchContent_GetProperties(vulkan_header_external)
if(NOT vulkan_header_external_POPULATED)
  FetchContent_Populate(vulkan_header_external)
  prep_thirdparty(${vulkan_header_external_SOURCE_DIR} "${vulkan_header_external_SOURCE_DIR}_fixed")
endif()
set(FIXED "${vulkan_header_external_SOURCE_DIR}_fixed")

add_library(VulkanHeaders_LIB INTERFACE)
add_dependencies(VulkanHeaders_LIB vulkan_header_external)
target_include_directories(VulkanHeaders_LIB SYSTEM INTERFACE ${FIXED}/include)
