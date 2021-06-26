file(GLOB SERIALIZABLE_SOURCES *.cpp *.h *.hpp)

add_library(SerializableLib STATIC ${SERIALIZABLE_SOURCES})

set_target_properties(  SerializableLib
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(SerializableLib PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(SerializableLib PRIVATE hashlib4cpp_external)
target_link_libraries(SerializableLib PUBLIC
    json_external
    glm_external
    CoreTools
    CoreDataStructures
)
