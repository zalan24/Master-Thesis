set(WARNINGFLAGS "${WARNINGFLAGS}   -Weverything \
                                    -Werror-return-type \
                                    -Wno-c++98-compat \
                                    -Wno-c++98-c++11-compat-binary-literal \
                                    -Wno-c++98-compat-pedantic \
                                    -Wno-float-equal \
                                    -Wno-reserved-id-macro \
                                    -Wno-unused-macros \
                                    -Wno-padded")
set(TEST_WARNINGFLAGS "${TEST_WARNINGFLAGS} -Werror-return-type \
                                            -Wno-c++98-compat -Wno-c++98-compat-pedantic \
                                            -Wall")