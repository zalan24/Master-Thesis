# This file aims to put all the mess of vindoz (and assimp) right under the carpet

function(prep_thirdparty directory outdir)
    if (NOT EXISTS ${outdir})
        message("Hacking 3rd party: ${directory}")
        file(REAL_PATH "${directory}/../" BASE_PATH)
        file(REMOVE_RECURSE "${BASE_PATH}/temp")
        file(RELATIVE_PATH FOLDER_PATH ${BASE_PATH} ${directory})
        file(COPY ${directory} DESTINATION "${BASE_PATH}/temp")
        file(RENAME "${BASE_PATH}/temp/${FOLDER_PATH}" ${outdir})
        file(REMOVE_RECURSE "${BASE_PATH}/temp")
        file(GLOB_RECURSE HEADERS "${outdir}/*.hpp" "${outdir}/*.h")
        file(GLOB_RECURSE SOURCES "${outdir}/*.c" "${outdir}/*.cpp" "${outdir}/*.cxx")
        foreach(F ${HEADERS})
            file(STRINGS ${F} CONTENT NEWLINE_CONSUME)
            string(PREPEND CONTENT
                "#pragma clang system_header\n"
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