#include "cmp_car_dis_to_cross_and_channel_id.h"
#include "car.h"

bool cmp_car_dis_to_cross_and_channel_id::operator() (const car &a, const car &b ){
    if (a.get_priority() == b.get_priority()) {
        if (a.get_dis_to_cross() == b.get_dis_to_cross())
            return a.get_channel_id() > b.get_channel_id();
        else
            return a.get_dis_to_cross() > b.get_dis_to_cross();
    } else
        return a.get_priority() < b.get_priority();
}