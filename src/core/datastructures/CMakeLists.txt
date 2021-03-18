file(GLOB DATA_STRUCTURES_SOURCES *.cpp *.h *.hpp)

# add_library(AnimationsDataStructures STATIC ${DATA_STRUCTURES_SOURCES})
add_library(AnimationsDataStructures INTERFACE ${DATA_STRUCTURES_SOURCES})
# set_target_properties(AnimationsDataStructures PROPERTIES LINKER_LANGUAGE CXX)

set_target_properties(  AnimationsDataStructures
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsDataStructures INTERFACE . ${CMAKE_CURRENT_BINARY_DIR})
