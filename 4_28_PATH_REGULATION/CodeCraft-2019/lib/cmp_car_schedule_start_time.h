#ifndef CMP_CAR_SCHEDULE_START_TIME
#define CMP_CAR_SCHEDULE_START_TIME

#include "car.h"

struct cmp_car_schedule_start_time {
public:
    bool operator() (const car &a, const car &b );
};

#endif