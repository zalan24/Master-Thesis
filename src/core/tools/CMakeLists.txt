file(GLOB CORE_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsTools STATIC ${CORE_SOURCES})
set_target_properties(AnimationsTools PROPERTIES LINKER_LANGUAGE CXX)

set_target_properties(  AnimationsTools
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsTools PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsTools PUBLIC
    glm_external
    boost_pp_external
    boost_type_traits_external
    boost_mpl_external
    boost_bind_external
    # flexible_reflection_external
    backward_cpp_external
    loguru_external
)
