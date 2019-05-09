#ifndef ROAD_H
#define ROAD_H

#include "config.h"
#include "car.h"
#include "cmp_car_dis_to_cross_and_channel_id.h"

#include <vector>
#include <list>
#include <queue>

using namespace std;

class road {
private:
    config my_config;
    int id; // road id
    int length; // road length
    int speed; // road max speed limit
    int channel; // road channel number
    int from; // road from cross id
    int to; // road to cross id
    // isDuplex == 1, road can from from_cross_id to to_cross_id
    // isDuplex == 1, road can from from_cross_id to to_cross_id and from to_cross_id to from_cross_id
    int is_duplex;
    
    // channel 0 to into_channel_id-1 can't into, initial value is 0 and into_channel_id < channel
    int into_channel_id;
    // size = 3, wait_into_road_direction_count[0] is straight, wait_into_road_direction_count[1] is left, wait_into_road_direction_count[2] is right
    vector<vector<int>> wait_into_road_direction_count;
    // dis_to_cross from small to large
    // if dis_to_cross equal, channel_id from small to large
    priority_queue<car, vector<car>, cmp_car_dis_to_cross_and_channel_id> wait_car_forefront_of_each_channel;
    
    // situation of cars in road
    // cars_in_road[channel_id] = list(car1 -> car2 -> car3 -> ...)
    // channel_id = [0, channel-1]
    // list<car> = car1->car2->car3->car4, ... the distance from the arrive_cross_id is from near to far
    vector<list<car>> cars_in_road;
    // cars wait to run in road
    vector<list<car>> cars_wait_to_run_in_road;
    
    // situation of cars running in road
    vector<double> situation_car_running_in_road;
public:
    // road_info = (id,length,speed,channel,from,to,isDuplex)
    road(string road_info);
    int get_id() const; // return road id
    int get_length() const;
    int get_speed() const;
    int get_from() const; // return road from cross id
    int get_to() const; // return road to cross id
    int get_is_duplex() const; // return is_duplex
    // swap from and to
    void swap_from_to();
    
    // set.into_channel_id = into_channel_id
    void set_into_channel_id(int into_channel_id);
    // road.wait_road_direction_count = [0, 0, 0]
    void init_wait_into_road_direction_count();
    // road.wait_car_forefront_of_each_channel.clear();
    void clear_wait_car_forefront_of_each_channel();
    // road.cars_in_road.clear()
    void clear_cars_in_road();
    // clear cars wait to run in road
    void clear_cars_wait_to_run_in_road();
    
    // if no car need through cross in this road
    bool if_no_car_through_cross();
    // get car which have priority through cross, wait_car_forefront_of_each_channel
    car get_car_priority_through_cross();
    // have car through cross
    void have_car_through_cross(int channel_id);
    
    // wait_into_road_direction_count[car_direct] += 1
    void add_wait_into_road_direction_count(int car_priority, int car_direct);
    // wait_into_road_direction_count[car_direct] -= 1
    void sub_wait_into_road_direction_count(int car_priority, int car_direct);
    // check whether car_turn_direct can enter this road
    bool check_direct_priority(int car_priority, int car_turn_direct);
    // add car wait to run in road
    void add_car_wait_to_run_in_road(car car_wait_to_run);
    
    // Schedule cars which arrive schedule time or wait start
    // cars_wait_to_run_in_road -> cars_in_road
    // return cars_wait_run_to_running_N
    int schedule_cars_wait_run(int cars_wait_run_priority, int T);
    // init cars' schedule state to wait state which in channel
    // return number of cars in channel
    int init_cars_schedule_status_in_channel(int channel_id);
    // Schedule cars in road.
    // If car can through cross then car into schedule wait -> car.schedule_status = 1
    // If car blocked by schedule wait car then car into schedule wait -> car.schedule_status = 1
    // If car don't be block and can't through cross then car run one time slice and into end state -> car.schedule_status = 0
    // If car blocked by termination state car then car move to the back of the previous car -> car.schedule_status = 0
    // return number of from wait state to termination status
    int schedule_cars_running_in_channel(int channel_id);
    int schedule_cars_running_in_road();
    // forefront car because some reason it need remain in cross
    int forefront_car_remain_in_cross(int channel_id);
    
    // check the road all channel whether be fill up
    // be fill up -> true else -> false
    bool whether_be_fill_up();
    // if car speed_in_road <= car.dis_to_cross in previous road, then dis_move_in_road = 0 -> car don't enter this road, car.dis_to_cross = 0 and return -1 
    // else if car into road don't be block or block by a car which is termination status, car enter road, return 1
    // else car can't enter road, need wait previous car to be termination state return 0
    int car_into_road(car into_car);
    
    // get car in road and schedule status is 1(wait status)
    vector<car> get_deadlock_car();
    
    // output road status
    void output_status(int T);
    
    //========================================================================
    
    // initial situation_car_running_in_road;
    void init_situation_car_running_in_road();
    
    // check whether cars can run into road between x to y
    double check_capacity(int x, int y, int car_speed);
    
    // modify the situation of cars running in road
    void car_running_count(int x, int y, double priority_weight);
    
    // count real situation car running in road during car schedule
    void count_real_situation_car_running_in_road(int T);
};

bool operator<(const road &a, const road &b);

#endif