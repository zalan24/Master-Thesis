file(GLOB DRV_COMMON_FILES *.cpp *.h *.hpp *.c)

add_library(DrvTypes STATIC ${DRV_COMMON_FILES})
target_include_directories(DrvTypes PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(  DrvTypes
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_link_libraries(DrvTypes PUBLIC
    # boost_stacktrace_external
    CoreDataStructures
    SerializableLib
    FeaturesLib
)
