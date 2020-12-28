file(GLOB IO_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsIO ${IO_SOURCES})

set_target_properties(  AnimationsIO
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS})
target_include_directories(AnimationsIO PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsIO PUBLIC AnimationsData AnimationsEntities
                                    PRIVATE ASSIMP_LIB BASE64_LIB)
