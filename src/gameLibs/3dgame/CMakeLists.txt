file(GLOB RENDERER_SOURCES *.cpp *.h *.hpp)

add_library(Game3D STATIC ${RENDERER_SOURCES})

set_target_properties(  Game3D
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(Game3D PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(Game3D PUBLIC
    AnimationsIRenderer
    AnimationsISimulation
    ShaderManager
    AnimationsEngine
    EngineRenderPass
)
