file(GLOB RUNTIME_STATS_SOURCES *.cpp *.h *.hpp)

add_executable(RuntimeStats ${RUNTIME_STATS_SOURCES})

set_target_properties(  RuntimeStats
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(RuntimeStats PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(RuntimeStats PUBLIC AnimationsIO AnimationFeatures)
