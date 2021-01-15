file(GLOB DRV_PROFILER_FILES *.cpp *.h *.hpp *.c)

add_library(ProjectiveGeometryDrvProfiler ${DRV_PROFILER_FILES})
target_include_directories(ProjectiveGeometryDrvProfiler PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(  ProjectiveGeometryDrvProfiler
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS})
set_target_properties(  ProjectiveGeometryDrvProfiler
                        PROPERTIES
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")