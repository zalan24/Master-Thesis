include(FetchContent)

option(BFD_ENABLE "Enable bdf (binutils-dev) for backward cpp" OFF)
option(DF_ENABLE "Enable dw (libdw-dev) for backward cpp" OFF)
option(DWARF_ENABLE "Enable dward (libdwarf-dev) for backward cpp" OFF)

configure_file(backwardcppconfig.h.in backwardcppconfig.h)

FetchContent_Declare(
    backtardcpp_external
    GIT_REPOSITORY https://github.com/bombela/backward-cpp.git
    GIT_TAG        v1.4
)

FetchContent_GetProperties(backtardcpp_external)
if(NOT backtardcpp_external_POPULATED)
  FetchContent_Populate(backtardcpp_external)
endif()
add_library(BackwardCpp_LIB INTERFACE)
add_dependencies(BackwardCpp_LIB backtardcpp_external)
target_include_directories(BackwardCpp_LIB SYSTEM INTERFACE
  ${backtardcpp_external_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR})

if(${BFD_ENABLE})
  target_link_libraries(BackwardCpp_LIB INTERFACE bfd dl)
endif()
if(${DF_ENABLE})
  # TODO
endif()
if(${DWARF_ENABLE})
  # TODO
endif()