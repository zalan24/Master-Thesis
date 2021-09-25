file(GLOB PHYSICS_SOURCES *.cpp *.h *.hpp)

add_library(PhysicsLib STATIC ${PHYSICS_SOURCES})

set_target_properties(  PhysicsLib
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(PhysicsLib PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(PhysicsLib PRIVATE
    bullet3_external
)
target_link_libraries(PhysicsLib PUBLIC
    glm_external
)
