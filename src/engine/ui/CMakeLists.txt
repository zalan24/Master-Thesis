file(GLOB WINDOW_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsUI STATIC ${WINDOW_SOURCES})

set_target_properties(  AnimationsUI
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsUI PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsUI PRIVATE GLAD_LIB AnimationsAssetLoaders)
