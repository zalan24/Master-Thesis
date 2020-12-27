file(GLOB TOOLS_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsTools ${TOOLS_SOURCES})
set_target_properties(AnimationsTools PROPERTIES LINKER_LANGUAGE CXX)

set_target_properties(  AnimationsTools
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS})
target_include_directories(AnimationsTools PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsTools PUBLIC Animations_GLM)
