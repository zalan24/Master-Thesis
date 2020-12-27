file(GLOB WINDOW_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsWindow ${WINDOW_SOURCES})

set_target_properties(  AnimationsWindow
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS})
target_include_directories(AnimationsWindow PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsWindow PRIVATE Animations_GLFW Animations_GLAD Animations_GLM AnimationsInput Animations_Imgui AnimationsIO)
target_link_libraries(AnimationsWindow PUBLIC AnimationsRenderer)
