file(GLOB DATA_STRUCTURES_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsDataStructures STATIC ${DATA_STRUCTURES_SOURCES})

set_target_properties(  AnimationsDataStructures
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsDataStructures PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
