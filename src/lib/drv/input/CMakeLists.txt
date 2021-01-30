file(GLOB INPUT_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsDrvInput STATIC ${INPUT_SOURCES})

set_target_properties(  AnimationsDrvInput
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsDrvInput PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsDrvInput PUBLIC CONCURRENT_QUEUE_LIB)
