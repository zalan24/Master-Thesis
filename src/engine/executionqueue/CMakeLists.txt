file(GLOB ENGINE_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsExecutionQueue STATIC ${ENGINE_SOURCES})

set_target_properties(  AnimationsExecutionQueue
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsExecutionQueue PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsExecutionQueue PUBLIC
    concurrent_queue_external
    AnimationsDrv
)
