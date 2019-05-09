#include "cross.h"

#include "road.h"
#include "util.h"

#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

cross::cross() {
}

// convert cross_info = (id,roadId1,roadId2,roadId3,roadId4) - > 
// turn_direct[roadId1][roadId2] = 1     turn left 
// turn_direct[roadId1][roadId3] = 0     straight
// turn_direct[roadId1][roadId3] = 2     turn right
// ...
cross::cross(string cross_info) {
    // road_info = (id,length,speed,channel,from,to,isDuplex)
    vector<int> info_val = parse_string_to_int_vector(cross_info);
    if (info_val.size() != 5)
        cout << "cross::cross error!!!!!!!!!!!!!!!!!" << endl;
    
    this->id = info_val[0];
    this->roads_into_cross.clear();
    this->roads_departure_cross.clear();
    // generate turn direct
    this->turn_direct.clear();
    this->road_id_in_cross.clear();
    for (int i = 0; i < 4; i ++) {
        info_val[i] = info_val[i + 1];
        if (info_val[i] != -1)
            this->road_id_in_cross.push_back(info_val[i]);
    }
    for (int i = 0; i < 4; i ++) {
        for (int j = 0; j < 4; j ++) {
            this->turn_direct[info_val[i]][info_val[j]] = -1;
        }
    }
    for (int i = 0; i < 4; i ++) {
        this->turn_direct[info_val[i]][info_val[(i + 1) % 4]] = 1;
        this->turn_direct[info_val[i]][info_val[(i + 2) % 4]] = 0;
        this->turn_direct[info_val[i]][info_val[(i + 3) % 4]] = 2;
    }
}

// return cross id
int cross::get_id() const {
    return this->id;
}

// query in this cross from from_road_id to to_road_id is 0->straight? 1->left? or 2->right?
int cross::get_turn_direct(int from_road_id, int to_road_id) {
    return this->turn_direct[from_road_id][to_road_id];
}

// add road* to road_into_cross which road->to = cross_id
void cross::add_road_into_cross(road* road_pointer) {
    this->roads_into_cross.push_back(road_pointer);
}

// add road* to road_departure_cross which road->from == cross_id
void cross::add_road_departure_cross(int road_id, road* road_pointer) {
    this->roads_departure_cross[road_id] = road_pointer;
}

// get road departure cross;
map<int, road*> cross::get_road_departure_cross() {
    return this->roads_departure_cross;
}

// get roads into cross
vector<road*> cross::get_roads_into_cross() {
    return this->roads_into_cross;
}

// add car wait to run in road
void cross::add_car_wait_to_run_in_road(car car_wait_to_run) {
    this->roads_departure_cross[car_wait_to_run.get_next_road_in_path()]->add_car_wait_to_run_in_road(car_wait_to_run);
}

// update road state in cross
// if for a road have 1 straight, 1 left then road.wait_into_road_direction_count = [1, 1, 0]
void cross::update_road_state_in_cross(road* road_pointer) {
    int now_road_id = road_pointer->get_id();
    if (road_pointer->if_no_car_through_cross())
        return;
    car car_through_cross = road_pointer->get_car_priority_through_cross();
    int next_road_id = car_through_cross.get_next_road_in_path();
    if (next_road_id == -1) {
        int count = 0;
        for (map<int, road*>::iterator iter = this->roads_departure_cross.begin(); iter != this->roads_departure_cross.end(); iter ++) {
            next_road_id = iter->first;
            if (this->turn_direct[now_road_id][next_road_id] == 0) {
                if (next_road_id == now_road_id) {
                    cout << "cross::schedule_cars_in_cross error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
                }
                this->roads_departure_cross[next_road_id]->add_wait_into_road_direction_count(car_through_cross.get_priority(), 0);
                count ++;
            }
            if (count > 1) {
                cout << "cross::schedule_cars_in_cross error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
            }
        }
        // car will arrive destination
        return;
    } else {
        int car_direct = this->turn_direct[now_road_id][next_road_id];
        this->roads_departure_cross[next_road_id]->add_wait_into_road_direction_count(car_through_cross.get_priority(), car_direct);
    }
}

// update road state in cross
// if for a road have 1 straight, 1 left then road.wait_into_road_direction_count = [1, 1, 0]
void cross::update_road_state_in_cross() {
    for (vector<road*>::iterator iter = roads_into_cross.begin(); iter != roads_into_cross.end(); iter ++) {
        this->update_road_state_in_cross(*iter);
    }
}

// Schedule cars which arrive schedule time or wait start
// cars_wait_schedule_start_time_list -> cars_wait_run_list
// cars_wait_run_list -> cars_in_road
void cross::schedule_cars_wait_run(int cars_wait_run_priority, vector<int> &cars_running_n, vector<int> &cars_wait_schedule_start_time_n, list<road> &roads_connect_cross, int T) {
    for (list<road>::iterator iter = roads_connect_cross.begin(); iter != roads_connect_cross.end(); iter ++) {
        int cars_wait_run_to_running_N = iter->schedule_cars_wait_run(cars_wait_run_priority, T);
        cars_wait_schedule_start_time_n[cars_wait_run_priority] -= cars_wait_run_to_running_N;
        cars_running_n[cars_wait_run_priority] += cars_wait_run_to_running_N;
    }
}

// schedule cars wait run in road
void cross::schedule_cars_wait_run_in_road(int cars_wait_run_priority, vector<int> &cars_running_n, vector<int> &cars_wait_schedule_start_time_n, road* road_to_run, int T) {
    int cars_wait_run_to_running_N = road_to_run->schedule_cars_wait_run(cars_wait_run_priority, T);
    cars_wait_schedule_start_time_n[cars_wait_run_priority] -= cars_wait_run_to_running_N;
    cars_running_n[cars_wait_run_priority] += cars_wait_run_to_running_N;
}

// car to next road, maybe return -2, -1, 0, 1
// if road is be termination status cars fill up return -2
// if car speed_in_road <= car.dis_to_cross in previous road, then dis_move_in_road = 0 -> car don't enter this road, car.dis_to_cross = 0 and return -1 
// else if car into road don't be block or block by a car which is termination status, car enter road, return 1
// else car can't enter road, need wait previous car to be termination state return 0
int cross::car_to_next_road(car car_through_cross) {
    road* next_road = this->roads_departure_cross[car_through_cross.get_next_road_in_path()];
    if (next_road->whether_be_fill_up())
        return -2;
    return next_road->car_into_road(car_through_cross);
}

// schedule cars can through cross
// return number of cars from wait status to termination status in this schedule
int cross::schedule_cars_in_cross(vector<int> &cars_running_n, vector<int> &cars_arrive_destination_n, vector<long long> &all_cars_running_time, int T, map<int, int> &arrive_car_id_count, vector<int> &arrive_T, vector<int> &cars_wait_schedule_start_time_n, list<road> &roads_connect_cross) {
    int wait_to_termination_n = 0;
    for (vector<road*>::iterator iter = this->roads_into_cross.begin(); iter != this->roads_into_cross.end(); iter ++) {
        while (!(*iter)->if_no_car_through_cross()) {
            car car_through_cross = (*iter)->get_car_priority_through_cross();
            int next_road_id = -1;
            if (car_through_cross.get_next_road_in_path() == -1) {
                // car will arrive destination, check road if have related straight road, if don't have next_road_id = -1
                for (map<int, road*>::iterator iter1 = this->roads_departure_cross.begin(); iter1 != this->roads_departure_cross.end(); iter1 ++) {
                    int now_road_id = (*iter)->get_id();
                    if (this->turn_direct[now_road_id][iter1->first] == 0) {
                        if (iter1->first == now_road_id) {
                            cout << "cross::schedule_cars_in_cross error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
                        }
                        next_road_id = iter1->first;
                        
                    }
                }
                if (next_road_id != -1) {
                    road* next_road = this->roads_departure_cross[next_road_id];
                    if (!next_road->check_direct_priority(car_through_cross.get_priority(), 0)) {
                        break;
                    }
                }
                int priority_val = car_through_cross.get_priority();
                wait_to_termination_n ++;
                cars_running_n[priority_val] --;
                cars_arrive_destination_n[priority_val] ++;
                all_cars_running_time[priority_val] += T - car_through_cross.get_plan_time();
                if (cars_running_n[priority_val] == 0)
                    arrive_T[priority_val] = T;
                //if (car_through_cross.get_id() == 14062)
                    //cout << "" << car_through_cross.get_id() << " " << car_through_cross.get_schedule_start_time() << " " << T << "" << endl;
                if (arrive_car_id_count[car_through_cross.get_id()] != 0) {
                    cout << "cross::schedule_cars_in_cross error!!!!!!!!!!!!!!!!!!!!!!" << endl;
                }
                arrive_car_id_count[car_through_cross.get_id()] = T;
                int channel_id = car_through_cross.get_channel_id();
                (*iter)->have_car_through_cross(channel_id);
                wait_to_termination_n += (*iter)->schedule_cars_running_in_channel(channel_id);
                // update direct priority
                if (next_road_id != -1)
                    this->roads_departure_cross[next_road_id]->sub_wait_into_road_direction_count(car_through_cross.get_priority(), 0);
                this->update_road_state_in_cross(*iter);
                
                //this->schedule_cars_wait_run(1, cars_running_n, cars_wait_schedule_start_time_n, roads_connect_cross, T);
                
                this->schedule_cars_wait_run_in_road(1, cars_running_n, cars_wait_schedule_start_time_n, (*iter), T);
                /*
                if (next_road_id != -1) {
                    road* next_road = this->roads_departure_cross[next_road_id];
                    this->schedule_cars_wait_run_in_road(1, cars_running_n, cars_wait_schedule_start_time_n, next_road, T);
                }
                */
                continue;
            } else 
                next_road_id = car_through_cross.get_next_road_in_path();
            road* next_road = this->roads_departure_cross[next_road_id];
            int car_turn_direct = this->turn_direct[((*iter)->get_id())][next_road->get_id()];
            if (next_road->check_direct_priority(car_through_cross.get_priority(), car_turn_direct)) 
            {
                // car direct have priority enter to road
                int channel_id = car_through_cross.get_channel_id();
                if (next_road->whether_be_fill_up()) {
                    // road is be fill up by cars which is termination status
                    wait_to_termination_n += (*iter)->forefront_car_remain_in_cross(channel_id);
                    // update direct priority
                    next_road->sub_wait_into_road_direction_count(car_through_cross.get_priority(), car_turn_direct);
                    this->update_road_state_in_cross(*iter);
                    
                    //this->schedule_cars_wait_run(1, cars_running_n, cars_wait_schedule_start_time_n, roads_connect_cross, T);
                    this->schedule_cars_wait_run_in_road(1, cars_running_n, cars_wait_schedule_start_time_n, (*iter), T);
                    //this->schedule_cars_wait_run_in_road(1, cars_running_n, cars_wait_schedule_start_time_n, next_road, T);
                    continue;
                } else {
                    // road don't be fill up
                    int flag = next_road->car_into_road(car_through_cross);
                    if (flag == -1) {
                        // car speed in next raod < dis_to_cross, so car be remain in cross
                        wait_to_termination_n += (*iter)->forefront_car_remain_in_cross(channel_id);
                        // update direct priority
                        next_road->sub_wait_into_road_direction_count(car_through_cross.get_priority(), car_turn_direct);
                        this->update_road_state_in_cross(*iter);
                        
                        //this->schedule_cars_wait_run(1, cars_running_n, cars_wait_schedule_start_time_n, roads_connect_cross, T);
                        this->schedule_cars_wait_run_in_road(1, cars_running_n, cars_wait_schedule_start_time_n, (*iter), T);
                        //this->schedule_cars_wait_run_in_road(1, cars_running_n, cars_wait_schedule_start_time_n, next_road, T);
                        continue;
                    } else if (flag == 0) {
                        // car block by car which is wait state
                        break;
                    } else if (flag == 1) {
                        // car enter next road
                        wait_to_termination_n ++;
                        (*iter)->have_car_through_cross(channel_id);
                        wait_to_termination_n += (*iter)->schedule_cars_running_in_channel(channel_id);
                        // update direct priority
                        next_road->sub_wait_into_road_direction_count(car_through_cross.get_priority(), car_turn_direct);
                        this->update_road_state_in_cross(*iter);
                        
                        //this->schedule_cars_wait_run(1, cars_running_n, cars_wait_schedule_start_time_n, roads_connect_cross, T);
                        this->schedule_cars_wait_run_in_road(1, cars_running_n, cars_wait_schedule_start_time_n, (*iter), T);
                        //this->schedule_cars_wait_run_in_road(1, cars_running_n, cars_wait_schedule_start_time_n, next_road, T);
                        continue;
                    }
                }
            } else {
                // car direct have priority enter to road
                break;
            }
        }
    }
    return wait_to_termination_n;
}

void cross::show_car_next_road_status(car car_wait_run) {
    this->roads_departure_cross[car_wait_run.get_next_road_in_path()]->output_status(0);
}