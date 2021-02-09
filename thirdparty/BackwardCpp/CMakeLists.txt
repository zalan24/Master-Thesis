option(BFD_ENABLE "Enable bdf (binutils-dev) for backward cpp" OFF)
option(DF_ENABLE "Enable dw (libdw-dev) for backward cpp" OFF)
option(DWARF_ENABLE "Enable dward (libdwarf-dev) for backward cpp" OFF)

configure_file(backwardcppconfig.h.in backwardcppconfig.h)

include(external)

download_external(
    backward_cpp
    GIT_REPOSITORY https://github.com/bombela/backward-cpp.git
	  GIT_TAG        v1.4
)

add_library(backward_cpp_external INTERFACE)
target_include_directories(backward_cpp_external SYSTEM INTERFACE
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_BINARY_DIR}/3rdParty/backward_cpp/src
)

if(${BFD_ENABLE})
  target_link_libraries(backward_cpp_external INTERFACE bfd dl)
endif()
if(${DF_ENABLE})
  # TODO
endif()
if(${DWARF_ENABLE})
  # TODO
endif()
