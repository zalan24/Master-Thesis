file(GLOB DRV_COMMON_FILES *.cpp *.h *.hpp *.c)

add_library(DrvCommon STATIC ${DRV_COMMON_FILES})
target_include_directories(DrvCommon PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(  DrvCommon
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_link_libraries(DrvCommon PUBLIC
    # boost_stacktrace_external
    DrvTypes
    RuntimeStats
    CoreDataStructures
    DrvConfigs
    AnimationsDrvProfiler
    CoreTools
    SerializableLib
    FeaturesLib
    Logging
    InputLib
)
