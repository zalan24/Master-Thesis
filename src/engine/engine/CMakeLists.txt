file(GLOB ENGINE_SOURCES *.cpp *.h *.hpp)

add_library(EngineLib STATIC ${ENGINE_SOURCES})

set_target_properties(  EngineLib
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(EngineLib PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(EngineLib PRIVATE
    json_external
)
target_link_libraries(EngineLib PUBLIC
    SerializableLib
    # AnimationsAssetLoaders
    AnimationsDrv
    # AnimationsResources
    InputLib
    EngineCmdBuffer
    EngineOneTimeCmdBuffer
    ShaderManager
    EngineFramegraph
    AnimationsEventPool
    RuntimeStats
    concurrent_queue_external
    loguru_external
    Entities
    TimestampPool
    PerfMetrics
    ImGuiLib
    EngineRenderPass
)
