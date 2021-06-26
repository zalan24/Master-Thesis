file(GLOB CORE_SOURCES *.cpp *.h *.hpp)

add_library(Logging STATIC ${CORE_SOURCES})
set_target_properties(Logging PROPERTIES LINKER_LANGUAGE CXX)

set_target_properties(  Logging
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(Logging PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(Logging PUBLIC
    loguru_external
)
