#include "cmp_car_id.h"
#include "car.h"

bool cmp_car_id::operator() (const car &a, const car &b ){
        return a.get_id() > b.get_id();
}