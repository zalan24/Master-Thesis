file(GLOB RENDER_PASS_SOURCES *.cpp *.h *.hpp)

add_library(EngineRenderPass STATIC ${RENDER_PASS_SOURCES})

set_target_properties(  EngineRenderPass
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(EngineRenderPass PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(EngineRenderPass PUBLIC
    AnimationsDrv
    AnimationsShaderManager
    RuntimeStats
)
