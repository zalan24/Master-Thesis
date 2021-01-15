file(GLOB ENTITIES_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsEntities ${ENTITIES_SOURCES})

set_target_properties(  AnimationsEntities
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsEntities PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsEntities PUBLIC GLM_LIB AnimationsSerializable)
