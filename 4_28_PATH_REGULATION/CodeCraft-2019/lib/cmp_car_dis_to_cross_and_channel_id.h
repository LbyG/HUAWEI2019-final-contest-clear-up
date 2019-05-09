#ifndef CMP_CAR_DIS_TO_CROSS_AND_CHANNEL_ID
#define CMP_CAR_DIS_TO_CROSS_AND_CHANNEL_ID

#include "car.h"

struct cmp_car_dis_to_cross_and_channel_id {
public:
    bool operator() (const car &a, const car &b );
};

#endif