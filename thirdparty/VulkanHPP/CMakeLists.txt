include(FetchContent)

FetchContent_Declare(
    vulkan_external
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Hpp.git
)

FetchContent_GetProperties(vulkan_external)
if(NOT vulkan_external_POPULATED)
  FetchContent_Populate(vulkan_external)
  prep_thirdparty(${vulkan_external_SOURCE_DIR} "${vulkan_external_SOURCE_DIR}_fixed")
endif()
set(FIXED "${vulkan_external_SOURCE_DIR}_fixed")

add_library(Vulkan_LIB INTERFACE)
add_dependencies(Vulkan_LIB vulkan_external)
target_include_directories(Vulkan_LIB SYSTEM INTERFACE ${FIXED})
