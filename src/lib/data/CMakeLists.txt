file(GLOB DATA_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsData ${DATA_SOURCES})

set_target_properties(  AnimationsData
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsData PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsData PUBLIC
    GLM_LIB
    AnimationsTools
    AnimationsSerializable
)
