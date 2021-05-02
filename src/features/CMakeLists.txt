set(FILENAME "features${CMAKE_BUILD_TYPE}.h")

add_library(FeaturesLib INTERFACE
    Debug/features.h
    Release/features.h
    featuresStructures.h
)

set_target_properties(  FeaturesLib
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")

target_include_directories(FeaturesLib INTERFACE . ${CMAKE_BUILD_TYPE} ${CMAKE_CURRENT_BINARY_DIR})
