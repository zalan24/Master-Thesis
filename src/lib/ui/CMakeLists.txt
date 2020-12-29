file(GLOB WINDOW_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsUI ${WINDOW_SOURCES})

set_target_properties(  AnimationsUI
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS})
target_include_directories(AnimationsUI PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsUI PRIVATE GLFW_LIB GLAD_LIB AnimationsInput AnimationsIO)
