file(GLOB DRV_MEMORY_FILES *.cpp *.h *.hpp *.c)

add_library(AnimationsDrvMemory STATIC ${DRV_MEMORY_FILES})
target_include_directories(AnimationsDrvMemory PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(AnimationsDrvMemory PUBLIC
    AnimationsDrvCommon
)
set_target_properties(  AnimationsDrvMemory
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
