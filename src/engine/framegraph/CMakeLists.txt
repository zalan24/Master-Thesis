file(GLOB ENGINE_SOURCES *.cpp *.h *.hpp)

add_library(EngineFramegraph STATIC ${ENGINE_SOURCES})

set_target_properties(  EngineFramegraph
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(EngineFramegraph PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(EngineFramegraph PUBLIC
    concurrent_queue_external
    AnimationsDrv
    AnimationsEventPool
    AnimationsDataStructures
    boost_align_external
)
