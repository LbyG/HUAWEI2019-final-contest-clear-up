#ifndef CAR_INFLUENCE_TO_ROAD_H
#define CAR_INFLUENCE_TO_ROAD_H

#include "road.h"
#include "influence_to_road.h"

using namespace std;

class car_influence_to_road {
private:
    vector<influence_to_road> car_influence_to_road;
public:
    // init car_influence_to_road
    void init_car_influence_to_road();
    // add a influence to car_influence_to_road;
    void add_car_influence_to_road(road* road_iter, int start_time, int end_time, double val);
    // eliminate car influence to road
    void eliminate_car_influence_to_road();
};

#endif