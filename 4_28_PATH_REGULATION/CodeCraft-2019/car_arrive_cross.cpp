#include "car_arrive_cross.h"

using namespace std;

// Constructor function
car_arrive_cross::car_arrive_cross() {
}

// Constructor function
car_arrive_cross::car_arrive_cross(int cross_id, int from_road_id, int car_arrive_time) {
    this->cross_id = cross_id;
    this->from_road_id = from_road_id;
    this->car_arrive_time = car_arrive_time;
}

// get cross_id
int car_arrive_cross::get_cross_id() const {
    return this->cross_id;
}

// get from road id
int car_arrive_cross::get_from_road_id() const {
    return this->from_road_id;
}

// get car arrive time
int car_arrive_cross::get_car_arrive_time() const {
    return this->car_arrive_time;
}

bool operator<(const car_arrive_cross &a, const car_arrive_cross &b) {
    return a.get_car_arrive_time() > b.get_car_arrive_time();
}