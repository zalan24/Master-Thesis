file(GLOB IO_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsAssetLoaders STATIC ${IO_SOURCES})

set_target_properties(  AnimationsAssetLoaders
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsAssetLoaders PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsAssetLoaders PRIVATE
    ASSIMP_LIB
    STB_LIB
    base64_external
)
target_link_libraries(AnimationsAssetLoaders PUBLIC
    glm_external
    AnimationsData
    AnimationsTools
    AnimationsSerializable
)
