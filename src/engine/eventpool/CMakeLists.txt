file(GLOB EVENT_POOL_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsEventPool STATIC ${EVENT_POOL_SOURCES})

set_target_properties(  AnimationsEventPool
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsEventPool PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsEventPool PUBLIC
    AnimationsDrv
)
