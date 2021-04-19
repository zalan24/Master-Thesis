include(external)

download_external(
    gfxreconstruct
    GIT_REPOSITORY https://github.com/LunarG/gfxreconstruct.git
	GIT_TAG        v0.9.6
)

external(
    gfxreconstruct
    INSTALL ON
    TARGETS
    PROJECT_OPTIONS
)
