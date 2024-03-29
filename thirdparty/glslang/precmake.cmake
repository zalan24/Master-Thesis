macro(find_host_package)
    if ("${ARGN}" MATCHES ".*PythonInterp.*")
        find_package (Python3 COMPONENTS Interpreter Development REQUIRED)
        set(PYTHONINTERP_FOUND TRUE)
        set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

        set(PYTHON_VERSION_STRING ${Python3_VERSION})
        set(PYTHON_VERSION_MAJOR ${Python3_VERSION_MAJOR})
        set(PYTHON_VERSION_MINOR ${Python3_VERSION_MINOR})
        set(PYTHON_VERSION_PATCH ${Python3_VERSION_PATCH})
    else()
        find_package(${ARGN})
    endif()
endmacro()
