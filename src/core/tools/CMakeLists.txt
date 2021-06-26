file(GLOB CORE_SOURCES *.cpp *.h *.hpp)

add_library(CoreTools STATIC ${CORE_SOURCES})
set_target_properties(CoreTools PROPERTIES LINKER_LANGUAGE CXX)

set_target_properties(  CoreTools
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(CoreTools PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(CoreTools PUBLIC
    glm_external
    boost_pp_external
    boost_type_traits_external
    boost_mpl_external
    boost_bind_external
    # flexible_reflection_external
    backward_cpp_external
)
