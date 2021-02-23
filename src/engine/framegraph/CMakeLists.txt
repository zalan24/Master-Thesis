file(GLOB ENGINE_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsFramegraph STATIC ${ENGINE_SOURCES})

set_target_properties(  AnimationsFramegraph
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsFramegraph PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsFramegraph PUBLIC
    AnimationsDrv
    AnimationsExecutionQueue
)
