set(FILENAME "features${CMAKE_BUILD_TYPE}.h")

add_library(AnimationFeatures INTERFACE
    Debug/features.h
    Release/features.h
    featuresStructures.h
)

set_target_properties(  AnimationFeatures
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")

target_include_directories(AnimationFeatures INTERFACE . ${CMAKE_BUILD_TYPE} ${CMAKE_CURRENT_BINARY_DIR})
