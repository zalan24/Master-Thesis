find_package(Vulkan REQUIRED)

if (Vulkan_FOUND)
    file(GLOB DRV_VULKAN_FILES
        *.c *.cpp *.h *.hpp
        commands/*.c commands/*.cpp commands/*.h commands/*.hpp
        components/*.c components/*.cpp components/*.h components/*.hpp)

    add_library(AnimationsDrvVulkan ${DRV_VULKAN_FILES})
    target_include_directories(AnimationsDrvVulkan PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    # target_include_directories(AnimationsDrvVulkan PRIVATE
    #     ${Vulkan_INCLUDE_DIRS}
    # )
    set_target_properties(  AnimationsDrvVulkan
                            PROPERTIES
                            COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                            LINK_OPTIONS "${TARGET_LINK_FLAGS}")
    target_link_libraries(AnimationsDrvVulkan PUBLIC
        AnimationsDrvCommon
        AnimationsDrvMemory
    )
    target_link_libraries(AnimationsDrvVulkan PRIVATE
        ${Vulkan_LIBRARIES}
        Vulkan_LIB
        VulkanHeaders_LIB
    )
endif()
