file(GLOB WINDOW_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsVulkanWindow STATIC ${WINDOW_SOURCES})

set_target_properties(  AnimationsVulkanWindow
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsVulkanWindow PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsVulkanWindow PRIVATE GLFW_LIB)
target_link_libraries(AnimationsVulkanWindow PUBLIC AnimationsDrvCommon)

target_link_libraries(AnimationsVulkanWindow PRIVATE
    ${Vulkan_LIBRARIES}
    Vulkan_LIB
    VulkanHeaders_LIB
)
