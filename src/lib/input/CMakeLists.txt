file(GLOB INPUT_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsInput ${INPUT_SOURCES})

set_target_properties(  AnimationsInput
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS})
target_include_directories(AnimationsInput PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsInput PUBLIC Animations_GLM)
