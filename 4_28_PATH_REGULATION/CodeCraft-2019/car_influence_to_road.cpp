#include "road.h"
#include "influence_to_road.h"
#include "car_influence_to_road.h"

using namespace std;

// init car_influence_to_road
void car_influence_to_road::init_car_influence_to_road() {
    this->car_influence_to_road.clear();
}

// add a influence to car_influence_to_road;
void car_influence_to_road::add_car_influence_to_road(road* road_iter, int start_time, int end_time, double val) {
    this->car_influence_to_road.push_back(influence_to_road(road_iter, start_time, end_time, val));
}

// eliminate car influence to road
void car_influence_to_road::eliminate_car_influence_to_road() {
    for (vector<influence_to_road>::iterator iter = this->car_influence_to_road.begin(); iter != car_influence_to_road.end(); iter ++)
        iter->eliminate_influence();
}