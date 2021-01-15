#include "microprofiler.h"

#include <time.h>

#define MAX_NUM_MILESTONES 200

static unsigned short int numMilestones = 0;
static struct Milestone milestones[MAX_NUM_MILESTONES];
static int enabled = 1;

void profile_reset(void) {
    numMilestones = 0;
}

void profile_enable(int _enabled) {
    enabled = _enabled;
}

void profile_milestone(const char* name) {
    if (numMilestones == MAX_NUM_MILESTONES || !enabled)
        return;
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    milestones[numMilestones].name = name;
    milestones[numMilestones].sec = (unsigned long)t.tv_sec;
    milestones[numMilestones].nsec = (unsigned long)t.tv_nsec;
    numMilestones++;
}

unsigned int profile_get_milestones(unsigned int num, struct Milestone* _milestones) {
    if (!_milestones)
        return numMilestones;
    unsigned int i = 0;
    for (; i < num && i < numMilestones; ++i)
        _milestones[i] = milestones[i];
    return i;
}

unsigned long long int profile_difference(struct Milestone* a, struct Milestone* b) {
    unsigned long long int d = 1000000000;
    unsigned long long int diff = b->sec - a->sec;
    unsigned long b_nsec = b->nsec;
    if (a->nsec > b_nsec) {
        b_nsec += d;
        diff--;
    }
    diff *= d;
    diff += b_nsec - a->nsec;
    return diff;
}
