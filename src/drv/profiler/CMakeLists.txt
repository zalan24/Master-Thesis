file(GLOB DRV_PROFILER_FILES *.cpp *.h *.hpp *.c)

add_library(AnimationsDrvProfiler STATIC ${DRV_PROFILER_FILES})
target_include_directories(AnimationsDrvProfiler PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(  AnimationsDrvProfiler
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
