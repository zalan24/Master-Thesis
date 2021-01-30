add_library(AnimationsMath INTERFACE)
set_target_properties(AnimationsMath PROPERTIES LINKER_LANGUAGE CXX)

set_target_properties(  AnimationsMath
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsMath INTERFACE . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsMath INTERFACE AnimationsTools)
