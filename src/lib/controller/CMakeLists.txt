file(GLOB CONTROLLER_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsController STATIC ${CONTROLLER_SOURCES})

set_target_properties(  AnimationsController
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsController PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsController PUBLIC
    GLM_LIB
    AnimationsTools
    AnimationsSerializable
    AnimationsMath
)
