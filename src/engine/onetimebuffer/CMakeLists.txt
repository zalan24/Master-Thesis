file(GLOB RENDER_PASS_SOURCES *.cpp *.h *.hpp)

add_library(EngineOneTimeCmdBuffer INTERFACE ${RENDER_PASS_SOURCES})

set_target_properties(  EngineOneTimeCmdBuffer
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(EngineOneTimeCmdBuffer INTERFACE . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(EngineOneTimeCmdBuffer INTERFACE
    AnimationsDrv
    EngineFramegraph
    EngineCmdBuffer
)
