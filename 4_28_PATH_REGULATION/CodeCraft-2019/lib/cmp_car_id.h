#ifndef CMP_CAR_ID
#define CMP_CAR_ID

#include "car.h"

struct cmp_car_id {
public:
    bool operator() (const car &a, const car &b );
};

#endif