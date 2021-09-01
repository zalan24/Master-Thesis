file(GLOB INPUT_SOURCES *.cpp *.h *.hpp)

add_library(InputLib STATIC ${INPUT_SOURCES})

set_target_properties(  InputLib
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(InputLib PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(InputLib PUBLIC concurrent_queue_external)
