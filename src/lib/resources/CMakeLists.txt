file(GLOB RESOURCES_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsResources STATIC ${RESOURCES_SOURCES})

set_target_properties(  AnimationsResources
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsResources PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsResources PUBLIC
    AnimationsData
    AnimationsBuffer
    AnimationsIO
)
