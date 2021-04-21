set(FILENAME "features${CMAKE_BUILD_TYPE}.h")

file(COPY ${FILENAME} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(RENAME ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME} ${CMAKE_CURRENT_BINARY_DIR}/features.h)

add_library(AnimationFeatures INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/features.h)


set_target_properties(  AnimationFeatures
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")

target_include_directories(AnimationFeatures INTERFACE . ${CMAKE_CURRENT_BINARY_DIR})
