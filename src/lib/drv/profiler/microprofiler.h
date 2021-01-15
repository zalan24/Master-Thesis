#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

    struct Milestone
    {
        const char* name;
        unsigned long sec;
        unsigned long nsec;
    };

    void profile_reset(void);
    void profile_milestone(const char*);
    void profile_enable(int enabled);
    unsigned int profile_get_milestones(unsigned int num, struct Milestone* milestones);
    // b >= a
    // returns (b-a) in nanosec
    unsigned long long int profile_difference(struct Milestone* a, struct Milestone* b);

#ifdef MICROPROFILE_ENABLED

#    define PROFILE_RESET profile_reset()
#    define PROFILE_MILESTONE(a) profile_milestone(a)
#    define PROFILE_ENABLE(b) profile_enable(b)

#else

#    define PROFILE_RESET (void)(0)
#    define PROFILE_MILESTONE(a) (void)(a)
#    define PROFILE_ENABLE(b) (void)(b)

#endif

#ifdef __cplusplus
}
#endif
