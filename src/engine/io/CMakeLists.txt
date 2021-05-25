file(GLOB IO_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsIO STATIC ${IO_SOURCES})

set_target_properties(  AnimationsIO
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsIO PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsIO PRIVATE
    base64_external
)
target_link_libraries(AnimationsIO PUBLIC
    AnimationsTools
    SerializableLib
)
