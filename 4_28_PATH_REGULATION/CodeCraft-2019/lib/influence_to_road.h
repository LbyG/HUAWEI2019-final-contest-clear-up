#ifndef INFLUENCE_TO_ROAD_H
#define INFLUENCE_TO_ROAD_H

#include "road.h"

using namespace std;

class influence_to_road {
private:
    road* road_iter;
    int start_time;
    int end_time;
    double val;
public:
    // Constructor function
    influence_to_road();
    // Constructor function
    influence_to_road(road* road_iter, int start_time, int end_time, double val);
    // eliminate influence to road
    void eliminate_influence();
};

#endif