find_package(Vulkan REQUIRED)

if (Vulkan_FOUND)
    file(GLOB DRV_VULKAN_FILES
        *.c *.cpp *.h *.hpp
        commands/*.c commands/*.cpp commands/*.h commands/*.hpp
        components/*.c components/*.cpp components/*.h components/*.hpp)

    add_library(ProjectiveGeometryDrvVulkan ${DRV_VULKAN_FILES})
    target_include_directories(ProjectiveGeometryDrvVulkan PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_include_directories(ProjectiveGeometryDrvVulkan PRIVATE
        ${Vulkan_INCLUDE_DIRS}
    )
    set_target_properties(  ProjectiveGeometryDrvVulkan
                            PROPERTIES
                            COMPILE_FLAGS ${TARGETCOMPILEFLAGS})
    set_target_properties(  ProjectiveGeometryDrvVulkan
                            PROPERTIES
                            LINK_OPTIONS "${TARGET_LINK_FLAGS}")
    target_link_libraries(ProjectiveGeometryDrvVulkan PUBLIC
        ProjectiveGeometryDrvCommon
        ProjectiveGeometryDrvMemory
    )
    target_link_libraries(ProjectiveGeometryDrvVulkan PRIVATE
        ${Vulkan_LIBRARIES}
        Vulkan_LIB
    )
endif()