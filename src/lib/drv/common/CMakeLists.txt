file(GLOB DRV_COMMON_FILES *.cpp *.h *.hpp *.c)

add_library(ProjectiveGeometryDrvCommon ${DRV_COMMON_FILES})
target_include_directories(ProjectiveGeometryDrvCommon PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(  ProjectiveGeometryDrvCommon
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS})
set_target_properties(  ProjectiveGeometryDrvCommon
                        PROPERTIES
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_link_libraries(ProjectiveGeometryDrvCommon PUBLIC
    ProjectiveGeometryDrvProfiler
)