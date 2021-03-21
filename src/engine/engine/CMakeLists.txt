file(GLOB ENGINE_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsEngine STATIC ${ENGINE_SOURCES})

set_target_properties(  AnimationsEngine
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsEngine PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsEngine PRIVATE
    json_external
)
target_link_libraries(AnimationsEngine PUBLIC
    AnimationsEntities
    AnimationsSerializable
    AnimationsAssetLoaders
    AnimationsDrv
    AnimationsIRenderer
    AnimationsISimulation
    AnimationsResources
    AnimationsInput
    AnimationsShaderManager
    AnimationsFramegraph
    AnimationsExecutionQueue
    AnimationsEventPool
    concurrent_queue_external
    loguru_external
)
