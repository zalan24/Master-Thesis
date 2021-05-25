file(GLOB DRV_CONFIGS_FILES *.cpp *.h *.hpp *.c)

add_library(DrvConfigs STATIC ${DRV_CONFIGS_FILES})
target_include_directories(DrvConfigs PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(  DrvConfigs
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_link_libraries(DrvConfigs PUBLIC
    SerializableLib
)
