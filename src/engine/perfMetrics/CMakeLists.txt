file(GLOB PERF_METRICS_SOURCES *.cpp *.h *.hpp)


file(READ perfmetrics.css CSS_FILE)
file(READ perfmetrics.js JS_FILE)
file(READ perfmetrics.html HTML_FILE)
configure_file(perf_metrics.cpp.in perf_metrics.cpp @ONLY)

list(APPEND PERF_METRICS_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/perf_metrics.cpp)

add_library(PerfMetrics STATIC ${PERF_METRICS_SOURCES})

set_target_properties(  PerfMetrics
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(PerfMetrics PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(PerfMetrics PUBLIC
    SerializableLib
)
