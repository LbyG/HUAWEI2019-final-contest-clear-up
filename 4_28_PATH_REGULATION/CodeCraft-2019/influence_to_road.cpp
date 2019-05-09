#include "road.h"
#include "influence_to_road.h"

using namespace std;

// Constructor function
influence_to_road::influence_to_road() {
}

// Constructor function
influence_to_road::influence_to_road(road* road_iter, int start_time, int end_time, double val) {
    this->road_iter = road_iter;
    this->start_time = start_time;
    this->end_time = end_time;
    this->val = val;
}

// eliminate influence to road
void influence_to_road::eliminate_influence() {
    this->road_iter->car_running_count(this->start_time, this->end_time, -1 * this->val);
}