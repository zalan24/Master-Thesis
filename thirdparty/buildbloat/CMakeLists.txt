include(FetchContent)

if (CMAKE_GENERATOR MATCHES "Ninja")
  option(BUILD_PROFILE "Profile ninja build" ON)
else()
  option(BUILD_PROFILE "Profile ninja build" OFF)
endif()

if (BUILD_PROFILE)
  FetchContent_Declare(
      buildbloat_external
      GIT_REPOSITORY https://github.com/nico/buildbloat
  )

  FetchContent_GetProperties(buildbloat_external)
  if(NOT mapbox_externalPOPULATED)
    FetchContent_Populate(buildbloat_external)
  endif()

  find_package(PythonInterp REQUIRED)

  if (${PYTHONINTERP_FOUND})
    configure_file(buildprofile.cmd.in buildprofile.cmd @ONLY)
  endif()
endif()