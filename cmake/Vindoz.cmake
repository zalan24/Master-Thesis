# This file aims to put all the mess of vindoz (and assimp) right under the carpet

function(prep_thirdparty directory out_dir)
    if (NOT EXISTS ${out_dir})
        message("Hacking 3rd party: ${directory}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${directory} ${out_dir} COMMAND_ERROR_IS_FATAL ANY)
        file(GLOB_RECURSE HEADERS ${out_dir}/*.hpp ${out_dir}/*.h)
        file(GLOB_RECURSE SOURCES ${out_dir}/*.c ${out_dir}/*.cpp ${out_dir}/*.cxx)
        foreach(F ${HEADERS})
            file(STRINGS ${F} CONTENT NEWLINE_CONSUME)
            string(PREPEND CONTENT
                "#ifdef __clang__\n"
                "#  pragma clang system_header\n"
                "#endif\n"
            )
            file(WRITE ${F} ${CONTENT})
        endforeach()
        foreach(F ${SOURCES})
            file(STRINGS ${F} CONTENT NEWLINE_CONSUME)
            string(PREPEND CONTENT
                "#pragma GCC diagnostic ignored \"-Wall\"\n"
                "#pragma clang diagnostic ignored \"-Weverything\"\n"
            )
            file(WRITE ${F} ${CONTENT})
        endforeach()
    endif()
endfunction()
