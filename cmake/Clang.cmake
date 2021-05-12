set(WARNINGFLAGS "${WARNINGFLAGS}   -Weverything \
                                    -Werror-return-type \
                                    -Werror-parentheses \
                                    -Werror-defaulted-function-deleted \
                                    -Werror-uninitialized \
                                    -Werror-undef \
                                    -Werror-call-to-pure-virtual-from-ctor-dtor \
                                    -Werror-reorder-ctor \
                                    -Wno-c++98-compat \
                                    -Wno-c++98-c++11-compat-binary-literal \
                                    -Wno-c++98-compat-pedantic \
                                    -Wno-float-equal \
                                    -Wno-reserved-id-macro \
                                    -Wno-unused-macros \
                                    -Wno-date-time \
                                    -Wno-padded")
set(TEST_WARNINGFLAGS "${TEST_WARNINGFLAGS} -Werror-return-type \
                                            -Werror-defaulted-function-deleted \
                                            -Wno-c++98-compat -Wno-c++98-compat-pedantic \
                                            -Wall")