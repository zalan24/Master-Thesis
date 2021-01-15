add_subdirectory(profiler)
add_subdirectory(memory)
add_subdirectory(common)

option(BUILD_DRV_CPU "Build driver for cpu" ON)
option(BUILD_DRV_CUDA "Build driver for cuda" ON)
option(BUILD_DRV_VULKAN "Build driver for vulkan" ON)

configure_file(drivers.h.in drivers.h @ONLY)

file(GLOB DRV_FILES *.cpp *.h *.hpp)
add_library(ProjectiveGeometryDRV ${DRV_FILES})
target_include_directories(ProjectiveGeometryDRV PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(  ProjectiveGeometryDRV
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS})
set_target_properties(  ProjectiveGeometryDRV
                        PROPERTIES
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_link_libraries(ProjectiveGeometryDRV PUBLIC
    ProjectiveGeometryDrvCommon
    ProjectiveGeometryDrvMemory
)

if (BUILD_DRV_CPU)
    add_subdirectory(cpu)
    target_link_libraries(ProjectiveGeometryDRV PRIVATE ProjectiveGeometryDrvCpu)
endif()
if (BUILD_DRV_CUDA)
    add_subdirectory(cuda)
    target_link_libraries(ProjectiveGeometryDRV PRIVATE ProjectiveGeometryDrvCuda)
endif()
if (BUILD_DRV_VULKAN)
    add_subdirectory(vulkan)
    target_link_libraries(ProjectiveGeometryDRV PRIVATE ProjectiveGeometryDrvVulkan)
endif()