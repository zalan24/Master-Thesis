file(GLOB DRV_COMMON_FILES *.cpp *.h *.hpp *.c drvtypes/*.cpp drvtypes/*.h drvtypes/*.hpp drvtypes/*.c)

add_library(AnimationsDrvCommon STATIC ${DRV_COMMON_FILES})
target_include_directories(AnimationsDrvCommon PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(  AnimationsDrvCommon
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_link_libraries(AnimationsDrvCommon PUBLIC
    AnimationsDrvProfiler
    AnimationsTools
    AnimationsSerializable
    AnimationFeatures
)