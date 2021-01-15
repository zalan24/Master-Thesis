enable_language(CUDA)

find_package(CUDA REQUIRED)

file(GLOB DRV_CUDA_FILES
    *.c *.cpp *.h *.hpp
    commands/*.c commands/*.cpp commands/*.h commands/*.hpp commands/*.cu
    components/*.c components/*.cpp components/*.h components/*.hpp components/*.cu
)

add_library(ProjectiveGeometryDrvCuda ${DRV_CUDA_FILES})
target_include_directories(ProjectiveGeometryDrvCuda PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(  ProjectiveGeometryDrvCuda
                        PROPERTIES
                        COMPILE_FLAGS ${TEST_TARGETCOMPILEFLAGS})
set_target_properties(  ProjectiveGeometryDrvCuda
                        PROPERTIES
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_link_libraries(ProjectiveGeometryDrvCuda PUBLIC
    ProjectiveGeometryDrvCommon
)
target_link_libraries(ProjectiveGeometryDrvCuda PRIVATE
    ${CUDA_LIBRARIES}
)
set_target_properties( ProjectiveGeometryDrvCuda
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)