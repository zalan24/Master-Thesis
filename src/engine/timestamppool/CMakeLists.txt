file(GLOB TIMESTAMP_POOL_SOURCES *.cpp *.h *.hpp)

add_library(TimestampPool STATIC ${TIMESTAMP_POOL_SOURCES})

set_target_properties(  TimestampPool
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(TimestampPool PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(TimestampPool PUBLIC
    AnimationsDrv
)
