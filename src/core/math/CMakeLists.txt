add_library(CoreMath INTERFACE)
set_target_properties(CoreMath PROPERTIES LINKER_LANGUAGE CXX)

set_target_properties(  CoreMath
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(CoreMath INTERFACE . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(CoreMath INTERFACE CoreTools)
