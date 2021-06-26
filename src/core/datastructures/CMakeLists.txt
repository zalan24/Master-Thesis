file(GLOB DATA_STRUCTURES_SOURCES *.cpp *.h *.hpp)

# add_library(CoreDataStructures STATIC ${DATA_STRUCTURES_SOURCES})
add_library(CoreDataStructures INTERFACE ${DATA_STRUCTURES_SOURCES})
# set_target_properties(CoreDataStructures PROPERTIES LINKER_LANGUAGE CXX)

set_target_properties(  CoreDataStructures
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(CoreDataStructures INTERFACE . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(CoreDataStructures INTERFACE
    Logging
)