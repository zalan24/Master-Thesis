file(GLOB RUNTIME_STATS_SOURCES *.cpp *.h *.hpp)

add_library(RuntimeStats STATIC ${RUNTIME_STATS_SOURCES})

set_target_properties(  RuntimeStats
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(RuntimeStats PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(RuntimeStats PUBLIC
    AnimationsDrv
    AnimationsIO
    FeaturesLib
)
