file(GLOB BUFFER_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsBuffer ${BUFFER_SOURCES})

set_target_properties(  AnimationsBuffer
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS})
target_include_directories(AnimationsBuffer PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsBuffer PUBLIC GLAD_LIB AnimationsTools AnimationsData)
