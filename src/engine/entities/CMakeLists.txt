file(GLOB ENTITIES_SOURCES *.cpp *.h *.hpp)

add_library(Entities STATIC ${ENTITIES_SOURCES})

set_target_properties(  Entities
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(Entities PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(Entities PUBLIC
    EngineFramegraph
    glm_external
    SerializableLib
)

