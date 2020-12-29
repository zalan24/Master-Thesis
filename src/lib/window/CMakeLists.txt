file(GLOB WINDOW_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsWindow ${WINDOW_SOURCES})

set_target_properties(  AnimationsWindow
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS})
target_include_directories(AnimationsWindow PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsWindow PRIVATE GLFW_LIB GLAD_LIB)
