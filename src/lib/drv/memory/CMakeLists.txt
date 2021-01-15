file(GLOB DRV_MEMORY_FILES *.cpp *.h *.hpp *.c)

add_library(ProjectiveGeometryDrvMemory ${DRV_MEMORY_FILES})
target_include_directories(ProjectiveGeometryDrvMemory PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(ProjectiveGeometryDrvMemory PUBLIC
    ProjectiveGeometryDrvCommon
)
set_target_properties(  ProjectiveGeometryDrvMemory
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS})
set_target_properties(  ProjectiveGeometryDrvMemory
                        PROPERTIES
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")