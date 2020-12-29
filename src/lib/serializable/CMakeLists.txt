file(GLOB SERIALIZABLE_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsSerializable ${SERIALIZABLE_SOURCES})

set_target_properties(  AnimationsSerializable
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS})
target_include_directories(AnimationsSerializable PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsSerializable PUBLIC
    JSON_LIB
)
