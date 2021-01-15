file(GLOB DRV_CPU_FILES
    *.c *.cpp *.h *.hpp
    components/*.c components/*.cpp components/*.h components/*.hpp
)

add_library(ProjectiveGeometryDrvCpu ${DRV_CPU_FILES})
target_include_directories(ProjectiveGeometryDrvCpu PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(  ProjectiveGeometryDrvCpu
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS})
set_target_properties(  ProjectiveGeometryDrvCpu
                        PROPERTIES
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_link_libraries(ProjectiveGeometryDrvCpu PUBLIC
    ProjectiveGeometryDrvCommon
    ProjectiveGeometryDrvMemory
)