# file(GLOB BUFFER_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsBuffer INTERFACE)

set_target_properties(  AnimationsBuffer
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS})
target_include_directories(AnimationsBuffer INTERFACE . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsBuffer INTERFACE GLAD_LIB AnimationsTools)
