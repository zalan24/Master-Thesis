file(GLOB INPUT_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsInput STATIC ${INPUT_SOURCES})

set_target_properties(  AnimationsInput
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsInput PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsInput PUBLIC concurrent_queue_external)
