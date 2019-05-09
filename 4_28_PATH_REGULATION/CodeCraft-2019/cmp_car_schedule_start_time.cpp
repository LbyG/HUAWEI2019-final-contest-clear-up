#include "cmp_car_schedule_start_time.h"
#include "car.h"

bool cmp_car_schedule_start_time::operator() (const car &a, const car &b ){
        return a.get_schedule_start_time() > b.get_schedule_start_time();
}