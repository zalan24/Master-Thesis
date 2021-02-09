file(GLOB SERIALIZABLE_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsSerializable STATIC ${SERIALIZABLE_SOURCES})

set_target_properties(  AnimationsSerializable
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsSerializable PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsSerializable PUBLIC
    json_external
    glm_external
)
