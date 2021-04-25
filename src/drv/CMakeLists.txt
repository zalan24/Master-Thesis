add_subdirectory(profiler)
add_subdirectory(configs)
add_subdirectory(common)

option(BUILD_DRV_VULKAN "Build driver for vulkan" ON)

configure_file(drivers.h.in drivers.h @ONLY)

file(GLOB DRV_FILES *.cpp *.h *.hpp)
add_library(AnimationsDrv STATIC ${DRV_FILES})
target_include_directories(AnimationsDrv PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(  AnimationsDrv
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_link_libraries(AnimationsDrv PUBLIC
    AnimationsDrvCommon
    AnimationsTools
)

if (BUILD_DRV_VULKAN)
    add_subdirectory(vulkan)
    target_link_libraries(AnimationsDrv PRIVATE AnimationsDrvVulkan)
endif()