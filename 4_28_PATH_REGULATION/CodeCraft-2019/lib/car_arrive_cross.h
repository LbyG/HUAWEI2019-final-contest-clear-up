#ifndef CAR_ARRIVE_CROSS_H
#define CAR_ARRIVE_CROSS_H

using namespace std;

class car_arrive_cross {
private:
    int cross_id;
    int from_road_id;
    int car_arrive_time;
public:
    // Constructor function
    car_arrive_cross();
    // Constructor function
    car_arrive_cross(int cross_id, int from_road_id, int car_arrive_time);
    // get cross_id
    int get_cross_id() const;
    // get from road id
    int get_from_road_id() const;
    // get car arrive time
    int get_car_arrive_time() const;
};

bool operator<(const car_arrive_cross &a, const car_arrive_cross &b);

#endif