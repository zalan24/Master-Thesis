find_package(Vulkan REQUIRED)

if (Vulkan_FOUND)
    file(GLOB DRV_VULKAN_FILES
        *.c *.cpp *.h *.hpp
        commands/*.c commands/*.cpp commands/*.h commands/*.hpp
        components/*.c components/*.cpp components/*.h components/*.hpp)

    add_library(AnimationsDrvVulkan STATIC ${DRV_VULKAN_FILES})
    target_include_directories(AnimationsDrvVulkan PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_target_properties(  AnimationsDrvVulkan
                            PROPERTIES
                            COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                            LINK_OPTIONS "${TARGET_LINK_FLAGS}")
    target_link_libraries(AnimationsDrvVulkan PUBLIC
        AnimationsDrvCommon
        AnimationsInput
        AnimationsTools
        AnimationsDataStructures
        AnimationFeatures
    )
    target_link_libraries(AnimationsDrvVulkan PRIVATE
        ${Vulkan_LIBRARIES}
        vulkan_hpp_external
        vulkan_headers_external
        glfw_external
    )
    set(VulkanLayersFolder "${PROJECT_BINARY_DIR}/vulkanLayers")
    set(VULKAN_LAYER_CONFIGS
        ${VulkanLayersFolder}/VkLayer_khronos_validation.json
        ${VulkanLayersFolder}/VkLayer_gfxreconstruct.json
    )

    add_custom_command(OUTPUT ${VULKAN_LAYER_CONFIGS}
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_BINARY_DIR}/3rdParty/validation_layers/install/lib/ ${VulkanLayersFolder}
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_BINARY_DIR}/3rdParty/gfxreconstruct/install/bin/ ${VulkanLayersFolder}
        COMMENT "Creating vulkan layers folder"
    )

    add_custom_target(VulkanLayers DEPENDS ${VULKAN_LAYER_CONFIGS})
    add_dependencies(VulkanLayers validation_layers_external gfxreconstruct_external)
    add_dependencies(AnimationsDrvVulkan VulkanLayers)
endif()
