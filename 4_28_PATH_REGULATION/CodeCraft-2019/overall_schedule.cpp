#include "overall_schedule.h"

#include "config.h"
#include "util.h"
#include "car_arrive_cross.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

// return cars number
int overall_schedule::get_cars_n() {
    return this->cars.size();
}

//load cars, roads and crosses info from car_path, road_path and cross_path file
void overall_schedule::load_cars_roads_crosses(string car_path, string road_path, string cross_path) {
    // load cars information from car_path file
    // car_info = (id,from,to,speed,planTime)
    this->cars.clear();
    ifstream cars_info_file(car_path);
    string car_info;
    if (cars_info_file)
    {
        while (getline(cars_info_file, car_info)) {
            if (car_info.size() > 0 && car_info[0] != '#') {
                car new_car = car(car_info);
                this->cars.insert(pair<int, car>(new_car.get_id(), new_car));
                //this->cars[new_car.get_id()] = new_car;
            }
        }
    }
    cars_info_file.close();
    // load roads information from road_path file
    // road_info = (id,length,speed,channel,from,to,isDuplex)
    this->roads.clear();
    ifstream roads_info_file(road_path);
    string road_info;
    if (roads_info_file)
    {
        while (getline(roads_info_file, road_info)) {
            if (road_info.size() > 0 && road_info[0] != '#') {
                road new_road = road(road_info);
                this->roads.insert(pair<int, road>(new_road.get_id(), new_road));
                //this->roads[new_road.get_id()] = new_road;
            }
        }
    }
    roads_info_file.close();
    // load crosses information from cross_path file
    // cross_info = (id,roadId1,roadId2,roadId3,roadId4)
    this->crosses.clear();
    ifstream crosses_info_file(cross_path);
    string cross_info;
    if (crosses_info_file)
    {
        while (getline(crosses_info_file, cross_info)) {
            if (cross_info.size() > 0 && cross_info[0] != '#') {
                cross new_cross = cross(cross_info);
                this->crosses.insert(pair<int, cross>(new_cross.get_id(), new_cross));
                //this->crosses[new_cross.get_id()] = new_cross;
            }
        }
    }
    roads_info_file.close();
    // connect road info and cross info
    // because iter->first is id, so road add to  road_into_cross is from small to large by road_id;
    this->roads_connect_cross.clear();
    for (map<int, road>::iterator iter = this->roads.begin(); iter != this->roads.end(); ++iter) {
        // road from from_id to to_id connect to crosses
        this->roads_connect_cross.push_back(road(iter->second));
        road* from_to_road = &this->roads_connect_cross.back();
        map<int, cross>::iterator cross_iter;
        cross_iter = this->crosses.find(from_to_road->get_to());
        cross_iter->second.add_road_into_cross(from_to_road);
        //this->crosses[from_to_road->get_to()].add_road_into_cross(from_to_road);
        cross_iter = this->crosses.find(from_to_road->get_from());
        cross_iter->second.add_road_departure_cross(iter->first, from_to_road);
        //this->crosses[from_to_road->get_from()].add_road_departure_cross(iter->first, from_to_road);
        // if isDuplex == 1m road from to_id to from_id connect to crosses
        if (iter->second.get_is_duplex() == 1) {
            this->roads_connect_cross.push_back(road(iter->second));
            this->roads_connect_cross.back().swap_from_to();
            road* to_from_road = &this->roads_connect_cross.back();
            cross_iter = this->crosses.find(to_from_road->get_to());
            cross_iter->second.add_road_into_cross(to_from_road);
            //this->crosses[to_from_road->get_to()].add_road_into_cross(to_from_road);
            cross_iter = this->crosses.find(to_from_road->get_from());
            cross_iter->second.add_road_departure_cross(iter->first, to_from_road);
            //this->crosses[to_from_road->get_from()].add_road_departure_cross(iter->first, to_from_road);
        }
    }
    // count metric a and metric b
    this->count_metric_ab();
}

// load cars' schedule plan from answer_path
// schedule_info = [id, schedule_start_time, schedule_path1, schedule_path2, schedule_path3, schedule_path4, ...]
void overall_schedule::load_answer(string answer_path) {
    ifstream answer_info_file(answer_path);
    string answer_info;
    if (answer_info_file)
    {
        while (getline(answer_info_file, answer_info)) {
            if (answer_info.size() > 0 && answer_info[0] != '#') {
                vector<int> answer_vec = parse_string_to_int_vector(answer_info);
                map<int, car>::iterator iter = this->cars.find(answer_vec[0]);
                iter->second.set_schedule_path(answer_vec);
                //cars[answer_vec[0]].set_schedule_path(answer_vec);
            }
        }
    }
    answer_info_file.close();
}

// load preset answer
void overall_schedule::load_preset_answer(string preset_answer_path) {
    ifstream answer_info_file(preset_answer_path);
    string answer_info;
    if (answer_info_file)
    {
        while (getline(answer_info_file, answer_info)) {
            if (answer_info.size() > 0 && answer_info[0] != '#') {
                vector<int> answer_vec = parse_string_to_int_vector(answer_info);
                //this->cars[answer_vec[0]].set_schedule_path(answer_vec);
                map<int, car>::iterator iter = this->cars.find(answer_vec[0]);
                iter->second.set_schedule_path(answer_vec);
            }
        }
    }
    answer_info_file.close();
}

// initial all data in T = 0
void overall_schedule::initial_cars_state_in_T0() {
    this->T = 1;
    this->arrive_T = vector<int>(this->my_config.priority_N, 0);
    this->all_cars_running_time = vector<long long>(this->my_config.priority_N, 0);
    // number of cars which T < car.schedule_start_time
    this->cars_wait_schedule_start_time_n = vector<int>(this->my_config.priority_N, 0);
    // the number of cars which is running in road
    this->cars_running_n = vector<int>(this->my_config.priority_N, 0); 
    // the number of cars which is arrive destination
    this->cars_arrive_destination_n = vector<int>(this->my_config.priority_N, 0);
    
    // the number of cars wait to through cross or previous car is wait to through cross
    this->cars_running_wait_state_n = 0; 
    // the number of cars was run in this time unit
    this->cars_running_termination_state_n = 0;
    
    // initial road status and load cars wait to run in road
    vector<car> cars_tmp;
    for (map<int, car>::iterator iter = this->cars.begin(); iter != this->cars.end(); iter ++)
        if (iter->second.get_whether_preset() == 1 || iter->second.get_whether_finish_find_path() > 0)
            cars_tmp.push_back(iter->second);
    sort(cars_tmp.begin(), cars_tmp.end());
    for (list<road>::iterator iter = this->roads_connect_cross.begin(); iter != this->roads_connect_cross.end(); iter ++) {
        iter->clear_cars_wait_to_run_in_road();
        iter->clear_cars_in_road();
    }
    
    for (vector<car>::iterator iter = cars_tmp.begin(); iter != cars_tmp.end(); iter ++) {
        this->crosses[iter->get_from()].add_car_wait_to_run_in_road(*iter);
        this->cars_wait_schedule_start_time_n[iter->get_priority()] ++;
    }
}

// Schedule cars in road.
// If car can through cross then car into schedule wait -> car.schedule_status = 1
// If car blocked by schedule wait car then car into schedule wait -> car.schedule_status = 1
// If car don't be block and can't through cross then car run one time slice and into end state -> car.schedule_status = 0
// If car blocked by termination state car then car move to the back of the previous car -> car.schedule_status = 0
void overall_schedule::schedule_cars_running_in_road() {
    // schedule all cars' state which running in road
    this->cars_running_wait_state_n = 0;
    for (list<road>::iterator iter = this->roads_connect_cross.begin(); iter != this->roads_connect_cross.end(); iter ++) {
        // schedule cars in road
        this->cars_running_wait_state_n += iter->schedule_cars_running_in_road();
    }
    // update roads' state
    for (map<int, cross>::iterator iter = crosses.begin(); iter != crosses.end(); iter ++) {
        // update road state in cross
        iter->second.update_road_state_in_cross();
    }
}

// Schedule cars which arrive schedule time or wait start
// cars_wait_schedule_start_time_list -> cars_wait_run_list
// cars_wait_run_list -> cars_in_road
void overall_schedule::schedule_cars_wait_run(int cars_wait_run_priority) {
    for (list<road>::iterator iter = this->roads_connect_cross.begin(); iter != this->roads_connect_cross.end(); iter ++) {
        int cars_wait_run_to_running_N = iter->schedule_cars_wait_run(cars_wait_run_priority, this->T);
        this->cars_wait_schedule_start_time_n[cars_wait_run_priority] -= cars_wait_run_to_running_N;
        this->cars_running_n[cars_wait_run_priority] += cars_wait_run_to_running_N;
    }
}

// schedule cars in one time unit
// if deadblock return false
bool overall_schedule::schedule_cars_one_time_unit(int car_priority) {
    // Schedule cars in road.
    // If car can through cross then car into schedule wait -> car.schedule_status = 1
    // If car blocked by schedule wait car then car into schedule wait -> car.schedule_status = 1
    // If car don't be block and can't through cross then car run one time slice and into end state -> car.schedule_status = 2
    // If car blocked by end state car then car move to the back of the previous car
    this->schedule_cars_running_in_road();
    this->schedule_cars_wait_run(1);
    
    //this->output_schedule_status();
    
    // Cycling the relevant roads at each cross until all car are end state
    while (this->cars_running_wait_state_n > 0) {
    
        int wait_to_termination_n = 0;
        for (map<int, cross>::iterator iter = this->crosses.begin(); iter != this->crosses.end(); iter ++) {
            wait_to_termination_n += iter->second.schedule_cars_in_cross(this->cars_running_n, this->cars_arrive_destination_n, this->all_cars_running_time, this->T, this->arrive_car_id_count, this->arrive_T, this->cars_wait_schedule_start_time_n, this->roads_connect_cross);
        }
        if (wait_to_termination_n == 0) {
            return false;
        }
        this->cars_running_wait_state_n -= wait_to_termination_n;
    }
    
    // Schedule cars which arrive schedule time or wait start
    // cars_wait_schedule_start_time_list -> cars_wait_run_list
    // cars_wait_run_list -> cars_in_road
    for (int i = this->my_config.priority_N - 1; i >= 0; i --)
        this->schedule_cars_wait_run(i);
    
    //    this->output_schedule_status();
    this->T ++;
    return true;
}

int overall_schedule::schedule_cars(int car_priority) {
//    this->arrive_car_id_count.clear();
    // initial state
//    this->initial_cars_state_in_T0();
    // If all cars is arrive then break
    while (!whether_vector_zero(this->cars_wait_schedule_start_time_n) || !whether_vector_zero(this->cars_running_n)) {
        if (!this->schedule_cars_one_time_unit(car_priority)) {
            cout << "cars_running_wait_state_n = " << this->cars_running_wait_state_n << endl; 
            this->output_schedule_status();
            this->prevent_deadlock(this->T, car_priority);
            return -1;
        }
        //for (list<road>::iterator iter = this->roads_connect_cross.begin(); iter != this->roads_connect_cross.end(); iter ++)
        //    iter->count_real_situation_car_running_in_road(this->T);
    }
//    this->output_schedule_status();
    
    // merge priority == 0 and priority == 1
    this->all_cars_running_time[0] += this->all_cars_running_time[1];
    this->arrive_T[0] = max(this->arrive_T[0], this->arrive_T[1]);
    // T for priority cars is from early plan time of priority cars to all priority cars arrive destination
    int priority_cars_early_plan_time = 1e8;
    for (map<int, car>::iterator iter = this->cars.begin(); iter != this->cars.end(); iter ++) {
        if (iter->second.get_priority() == 1)
            priority_cars_early_plan_time = min(priority_cars_early_plan_time, iter->second.get_plan_time());
    }
    this->arrive_T[1] -= priority_cars_early_plan_time;
    
    for (int i = 0; i < this->my_config.priority_N; i ++)
        cout << "priority = " << i << " T = " << this->arrive_T[i] << " all cars running time = " << this->all_cars_running_time[i] << endl;
    cout << "metric_a = " << this-> metric_a << " metric_b = " << this->metric_b << endl;
    int final_arrive_T = round(this->arrive_T[0] + this->metric_a * this->arrive_T[1]);
    long long final_all_cars_running_time = round(this->all_cars_running_time[0] + this->metric_b * this->all_cars_running_time[1]);
    cout << "final result: " << " T = " << final_arrive_T << " all cars running time = " << final_all_cars_running_time << endl;
    return this->T;
}

// count metric a and metric b
void overall_schedule::count_metric_ab() {
    int min_val = -1e8;
    int max_val = 1e8;
    int cars_N = 0;
    int priority_cars_N = 0;
    int cars_fastest_speed = min_val;
    int cars_lowest_speed = max_val;
    int priority_cars_fastest_speed = min_val;
    int priority_cars_lowest_speed = max_val;
    int cars_later_plan_time = min_val;
    int cars_early_plan_time = max_val;
    int priority_cars_later_plan_time = min_val;
    int priority_cars_early_plan_time = max_val;
    int cars_from_count = 0;
    int priority_cars_from_count = 0;
    int cars_to_count = 0;
    int priority_cars_to_count = 0;
    map<int, int> cars_from_map;
    map<int, int> priority_cars_from_map;
    map<int, int> cars_to_map;
    map<int, int> priority_cars_to_map;
    for (map<int, car>::iterator iter = this->cars.begin(); iter != this->cars.end(); iter ++) {
        car iter_car = iter->second;
        cars_N ++;
        cars_fastest_speed = max(cars_fastest_speed, iter_car.get_speed());
        cars_lowest_speed = min(cars_lowest_speed, iter_car.get_speed());
        cars_later_plan_time = max(cars_later_plan_time, iter_car.get_plan_time());
        cars_early_plan_time = min(cars_early_plan_time, iter_car.get_plan_time());
        cars_from_map[iter_car.get_from()] ++;
        if (cars_from_map[iter_car.get_from()] == 1)
            cars_from_count ++;
        cars_to_map[iter_car.get_to()] ++;
        if (cars_to_map[iter_car.get_to()] == 1)
            cars_to_count ++;
        if (iter_car.get_priority() == 1) {
            priority_cars_N ++;
            priority_cars_fastest_speed = max(priority_cars_fastest_speed, iter_car.get_speed());
            priority_cars_lowest_speed = min(priority_cars_lowest_speed, iter_car.get_speed());
            priority_cars_later_plan_time = max(priority_cars_later_plan_time, iter_car.get_plan_time());
            priority_cars_early_plan_time = min(priority_cars_early_plan_time, iter_car.get_plan_time());
            priority_cars_from_map[iter_car.get_from()] ++;
            if (priority_cars_from_map[iter_car.get_from()] == 1)
                priority_cars_from_count ++;
            priority_cars_to_map[iter_car.get_to()] ++;
            if (priority_cars_to_map[iter_car.get_to()] == 1)
                priority_cars_to_count ++;
        }
    }
    long double metric_N = 1.0 * cars_N / priority_cars_N;
    long double metric_speed = 1.0 * (1.0 * cars_fastest_speed / cars_lowest_speed) / (1.0 * priority_cars_fastest_speed / priority_cars_lowest_speed);
    long double metric_plan_time = 1.0 * (1.0 * cars_later_plan_time / cars_early_plan_time) / (1.0 * priority_cars_later_plan_time / priority_cars_early_plan_time);
    long double metric_from = 1.0 * cars_from_count / priority_cars_from_count;
    long double metric_to = 1.0 * cars_to_count / priority_cars_to_count;
    metric_N = 1.0 * round(metric_N * 1e5) / 1e5;
    metric_speed = 1.0 * round(metric_speed * 1e5) / 1e5;
    metric_plan_time = 1.0 * round(metric_plan_time * 1e5) / 1e5;
    metric_from = 1.0 * round(metric_from * 1e5) / 1e5;
    metric_to = 1.0 * round(metric_to * 1e5) / 1e5;
    this->metric_a = 0.05 * metric_N + 0.2375 * metric_speed + 0.2375 * metric_plan_time + 0.2375 * metric_from + 0.2375 * metric_to;
    this->metric_b = 0.8 * metric_N + 0.05 * metric_speed + 0.05 * metric_plan_time + 0.05 * metric_from + 0.05 * metric_to;
    this->priority_weight = vector<long double>(2, 1);
    this->priority_weight[1] = this->metric_a;
    //this->metric_a = 1.0 * round(this->metric_a * 1e5) / 1e5;
    //this->metric_b = 1.0 * round(this->metric_b * 1e5) / 1e5;
}

// output car schedule status
void overall_schedule::output_schedule_status() {
    
    cout << "========show schedule status=========" << endl;
    cout << "T = " << this->T << endl;
    /*
    cout << "cars_wait_schedule_start_time_n = " << this->cars_wait_schedule_start_time_n << endl;
    cout << "cars_wait_run_n = " << this->cars_wait_run_n << endl;
    cout << "car is list which wait run: ";
    for (vector<car>::iterator iter = this->cars_wait_run_list.begin(); iter != this->cars_wait_run_list.end(); iter ++) {
        cout << iter->get_id() << " ";
    }
    cout << endl;
    cout << "cars_running_n = " << this->cars_running_n << endl;
    cout << "cars_arrive_destination_n = " << this->cars_arrive_destination_n << endl;
    cout << "cars_running_wait_state_n = " << this->cars_running_wait_state_n << endl;
    cout << "cars_running_termination_state_n = " << this->cars_running_termination_state_n << endl;
    cout << "car running in the road:" << endl;   
    for (list<road>::iterator iter = roads_connect_cross.begin(); iter != roads_connect_cross.end(); iter ++) {
     //   if (iter->get_id() == 5074)
     //   if (!iter->if_no_car_through_cross())
            iter->output_status(this->T);
    }
    */
    cout << "=====================================" << endl;
}

//================================================================
    
// count preset car to situation_car_running_in_road
void overall_schedule::preset_car_to_situation(car preset_car) {
    int schedule_start_time = preset_car.get_schedule_start_time();
    int cross_id = preset_car.get_from();
    while (preset_car.get_next_road_in_path() != -1) {
        int next_road_id = preset_car.get_next_road_in_path();
        road* next_road = this->crosses[cross_id].get_road_departure_cross()[next_road_id];
        int speed = min(preset_car.get_speed(), next_road->get_speed());
        int length = next_road->get_length();
        int through_time = (length + speed - 1) / speed;
        next_road->car_running_count(schedule_start_time, schedule_start_time + through_time - 1, this->priority_weight[preset_car.get_priority()]);
        schedule_start_time += through_time;
        cross_id = next_road->get_to();
        preset_car.arrive_next_road_path();
    }
    
}

// some way to prevent deadlock
void overall_schedule::prevent_deadlock(int T, int car_priority) {
    // find deadlock cycle road======================================================
    map<road*, int> deadlock_road_into_road_count;
    deadlock_road_into_road_count.clear();
    map<road*, road*> deadlock_road_into_road;
    for (list<road>::iterator iter = roads_connect_cross.begin(); iter != roads_connect_cross.end(); iter ++) {
        if (!iter->if_no_car_through_cross()) {
            car deadlock_car = iter->get_car_priority_through_cross();
            int now_road_id = iter->get_id();
            int next_road_id = deadlock_car.get_next_road_in_path();
            cross through_cross = this->crosses[iter->get_to()];
            road* now_road = NULL;
            vector<road*> roads_into_cross = through_cross.get_roads_into_cross();
            int flag = 0;
            for (vector<road*>::iterator iter = roads_into_cross.begin(); iter != roads_into_cross.end(); iter ++)
                if ((*iter)->get_id() == now_road_id) {
                    now_road = (*iter);
                    flag = 1;
                }
            if (flag == 0) {
                cout << "overall_schedule::prevent_deadlock error!!!!!!!!!!!!!!!!!!!!!!!" << endl;
            }
            if (next_road_id == -1)
                continue;
            road* next_road = through_cross.get_road_departure_cross()[next_road_id];
            deadlock_road_into_road[now_road] = next_road;
            deadlock_road_into_road_count[next_road] ++;
            //deadlock_road_into_road_count[now_road] = deadlock_road_into_road_count[now_road];
        }
    }
    list<road*> road_list;
    for (map<road*, int>::iterator iter = deadlock_road_into_road_count.begin(); iter != deadlock_road_into_road_count.end(); iter ++) {
        if (iter->second == 0)
            road_list.push_back(iter->first);
    }
    while (!road_list.empty()) {
        road* now_road = road_list.front();
        road_list.pop_front();
        if (deadlock_road_into_road[now_road] == 0)
            continue;
        road* next_road = deadlock_road_into_road[now_road];
        deadlock_road_into_road_count[next_road] --;
        if (deadlock_road_into_road_count[next_road] == 0)
            road_list.push_back(next_road);
    }
    // find deadlock cycle road======================================================
    
    int deadlock_road_count = 0;
    int count = 0;
    for (map<road*, int>::iterator iter = deadlock_road_into_road_count.begin(); iter != deadlock_road_into_road_count.end(); iter ++) {
        if (!(iter->first)->if_no_car_through_cross() && (iter->second != 0)) {
            deadlock_road_count ++;
            //iter->add_deadlock_situation_car_running_in_road(max(0, T - 100), T + 50, 1);
            vector<car> deadlock_car = iter->first->get_deadlock_car();
            int road_count = 0;
            for (vector<car>::iterator car_iter = deadlock_car.begin(); car_iter != deadlock_car.end(); car_iter ++) {
                if (car_iter->get_whether_preset() == 0) {
                    this->deadlock_epoch_step = (this->deadlock_epoch_step) % 9;
                    if (car_iter->get_priority() == car_priority && this->cars[car_iter->get_id()].get_whether_finish_find_path() == 1) {
                        this->cars[car_iter->get_id()].set_whether_finish_find_path(0);
                        count ++;
                        road_count ++;
                        if (road_count > this->my_config.max_withdraw_in_road)
                            break;
                    }
                }
            }
        }
    }
    cout << "deadlock count = " << count << endl;
    cout << "deadlock road count = " << deadlock_road_count << endl;
    if (count == 0) {
        for (vector<car>::iterator car_iter = this->cars_start_run.begin(); car_iter != this->cars_start_run.end(); car_iter ++) {
            if (this->cars[car_iter->get_id()].get_whether_finish_find_path() == 1) {
                if (count == 0)
                    this->cars[car_iter->get_id()].set_whether_finish_find_path(0);
                count ++;
                if (count == this->my_config.prevent_deadlock_metric)
                    count = 0;
            }
        }
    }
}

// estimate time car go to destination
int overall_schedule::estimate_spend_time(car estimate_car) {
    const int MAX_VAL = 1e8;
    map<int, int> cross_dis;
    map<int, int> cross_flag;
    typedef pair<int, int> QueueItem;
    priority_queue <QueueItem, std::vector<QueueItem>, std::greater<QueueItem> > cross_arrive_time_queue;
    for (map<int, cross>::iterator iter = this->crosses.begin(); iter != this->crosses.end(); iter ++) {
        int cross_id = iter->second.get_id();
        cross_dis[cross_id] = MAX_VAL;
        cross_flag[cross_id] = 0;
    }
    cross_dis[estimate_car.get_from()] = 0;
    cross_arrive_time_queue.push(pair<int, int>(cross_dis[estimate_car.get_from()], estimate_car.get_from()));
    while (true) {
        int update_cross_id = -1;
        int update_dis = MAX_VAL;
        while (!cross_arrive_time_queue.empty()) {
            if (cross_flag[cross_arrive_time_queue.top().second] == 0) {
                update_dis = cross_arrive_time_queue.top().first;
                update_cross_id = cross_arrive_time_queue.top().second;
                cross_arrive_time_queue.pop();
                break;
            }
            cross_arrive_time_queue.pop();
        }
        if (update_cross_id == -1)
            break;
        if (update_cross_id == estimate_car.get_to())
            return update_dis;
        cross_flag[update_cross_id] = 1;
        map<int, road*> road_departure_cross = this->crosses[update_cross_id].get_road_departure_cross();
        for (map<int, road*>::iterator road_iter = road_departure_cross.begin(); road_iter != road_departure_cross.end(); road_iter ++) {
            int to_cross_id = road_iter->second->get_to();
            int speed = min(road_iter->second->get_speed(), estimate_car.get_speed());
            int length = road_iter->second->get_length();
            int through_time = (length + speed - 1) / speed;
            if (cross_dis[to_cross_id] > update_dis + through_time) {
                cross_dis[to_cross_id] = update_dis + through_time;
                cross_arrive_time_queue.push(pair<int, int>(cross_dis[to_cross_id], to_cross_id));
            }
        }
    }
}

// reluate car start time and path
int overall_schedule::car_path_regulation(car car_iter, int start_time) {
    while (true) {
        vector<road*> roads_into_cross;
        map<int, map<int, int>> cross_arrive_time;
        map<int, map<int, bool>> updata_flag;
        map<int, map<int, int>> updata_from_cross_id;
        map<int, map<int, int>> updata_from_road_id;
        priority_queue <car_arrive_cross> cross_arrive_time_queue;
        for (map<int, cross>::iterator cross_iter = this->crosses.begin(); cross_iter != this->crosses.end(); cross_iter ++) {
            roads_into_cross = cross_iter->second.get_roads_into_cross();
            for (vector<road*>::iterator road_iter = roads_into_cross.begin(); road_iter != roads_into_cross.end(); road_iter ++) {
                cross_arrive_time[cross_iter->first][(*road_iter)->get_id()] = this->my_config.MAX_VAL;
                updata_flag[cross_iter->first][(*road_iter)->get_id()] = false;
                updata_from_cross_id[cross_iter->first][(*road_iter)->get_id()] = cross_iter->first;
                updata_from_road_id[cross_iter->first][(*road_iter)->get_id()] = 0;
            }
        }
        roads_into_cross = this->crosses[car_iter.get_from()].get_roads_into_cross();
        for (vector<road*>::iterator road_iter = roads_into_cross.begin(); road_iter != roads_into_cross.end(); road_iter ++) {
            cross_arrive_time[car_iter.get_from()][(*road_iter)->get_id()] = max(car_iter.get_plan_time(), start_time);
            cross_arrive_time_queue.push(car_arrive_cross(car_iter.get_from(), (*road_iter)->get_id(), cross_arrive_time[car_iter.get_from()][(*road_iter)->get_id()]));
        }
        while (true) {
            int updata_cross_id = -1;
            int updata_road_id = -1;
            int updata_cross_time = this->my_config.MAX_VAL;
            while (!cross_arrive_time_queue.empty()) {
                if (!updata_flag[cross_arrive_time_queue.top().get_cross_id()][cross_arrive_time_queue.top().get_from_road_id()]) {
                    updata_cross_id = cross_arrive_time_queue.top().get_cross_id();
                    updata_road_id = cross_arrive_time_queue.top().get_from_road_id();
                    updata_cross_time = cross_arrive_time_queue.top().get_car_arrive_time();
                    cross_arrive_time_queue.pop();
                    break;
                }
                cross_arrive_time_queue.pop();
            }
            if (updata_cross_id == -1 || updata_cross_id == car_iter.get_to())
                break;
            updata_flag[updata_cross_id][updata_road_id] = true;
            map<int, road*> roads_departure_cross = this->crosses[updata_cross_id].get_road_departure_cross();
            for (map<int, road*>::iterator road_iter = roads_departure_cross.begin(); road_iter != roads_departure_cross.end(); road_iter ++) {
                int next_road_id = (road_iter->second)->get_id();
                if (next_road_id == updata_road_id)
                    continue;
                int turn_direct = this->crosses[updata_cross_id].get_turn_direct(updata_road_id, next_road_id);
                int speed = min(car_iter.get_speed(), (road_iter->second)->get_speed());
                int length = (road_iter->second)->get_length();
                int through_time = (length + speed - 1) / speed + turn_direct * (1 - car_iter.get_priority()) * 0;
                int next_cross_id = (road_iter->second)->get_to();
                double congestion = road_iter->second->check_capacity(updata_cross_time, updata_cross_time + through_time - 1, car_iter.get_speed());
                if (congestion < 1) {
                    congestion = this->my_config.congestion_metric * max(congestion - this->my_config.congestion_threshold, 0.0);
                    if (cross_arrive_time[next_cross_id][next_road_id] > through_time * (1 + congestion) + cross_arrive_time[updata_cross_id][updata_road_id]) {
                        cross_arrive_time[next_cross_id][next_road_id] = through_time * (1 + congestion) + cross_arrive_time[updata_cross_id][updata_road_id];
                        updata_from_cross_id[next_cross_id][next_road_id] = updata_cross_id;
                        updata_from_road_id[next_cross_id][next_road_id] = updata_road_id;
                        cross_arrive_time_queue.push(car_arrive_cross(next_cross_id, next_road_id, through_time));
                        this->car_can_from_to[this->car_can_from_to_time][car_iter.get_from()][next_cross_id] = this->car_can_from_to_flag[this->car_can_from_to_time][car_iter.get_from()] + 1;
                    }
                }
            }
        }
        int min_cross_arrive_time = this->my_config.MAX_VAL;
        int road_id = -1;
        int cross_id = car_iter.get_to();
        roads_into_cross = this->crosses[cross_id].get_roads_into_cross();
        for (vector<road*>::iterator road_iter = roads_into_cross.begin(); road_iter != roads_into_cross.end(); road_iter ++) {
            if (cross_arrive_time[cross_id][(*road_iter)->get_id()] < min_cross_arrive_time) {
                min_cross_arrive_time = cross_arrive_time[cross_id][(*road_iter)->get_id()];
                road_id = (*road_iter)->get_id();
            }
        }
        if (road_id == -1)
            return -1;
        vector<int> schedule_path;
        schedule_path.clear();
        while (cross_id != car_iter.get_from()) {
            int next_cross_arrive_time = cross_arrive_time[cross_id][road_id];
            schedule_path.push_back(road_id);
            int pre_cross_id = updata_from_cross_id[cross_id][road_id];
            int pre_road_id = updata_from_road_id[cross_id][road_id];
            map<int, road*> roads_departure_cross = this->crosses[pre_cross_id].get_road_departure_cross();
            int pre_cross_arrive_time = cross_arrive_time[pre_cross_id][pre_road_id];
            roads_departure_cross[road_id]->car_running_count(pre_cross_arrive_time, next_cross_arrive_time, this->priority_weight[car_iter.get_priority()]);
            cross_id = pre_cross_id;
            road_id = pre_road_id;
        }
        vector<int> schedule_info;
        schedule_info.push_back(car_iter.get_id());
        schedule_info.push_back(start_time);
        for (vector<int>::reverse_iterator riter = schedule_path.rbegin();riter!=schedule_path.rend();riter++)
            schedule_info.push_back(*riter);
        //car_iter->set_schedule_path(schedule_info);
        this->cars[car_iter.get_id()].set_schedule_path(schedule_info);
        this->cars[car_iter.get_id()].set_whether_finish_find_path(1);
        break;
    }
    return 1;
}

void overall_schedule::modify_preset_car() {
    map<int, int> preset_car_schedule_start_time;
    for (list<road>::iterator iter = this->roads_connect_cross.begin(); iter != this->roads_connect_cross.end(); iter ++) {
        iter->init_situation_car_running_in_road();
    }
    // Adjust the order of the vehicle from low to high speed anc count preset car to situation_car_running_in_road
    this->cars_to_regulate_path = vector<list<car>>(this->my_config.priority_N, list<car>());
    vector<car> preset_car; 
    for (map<int, car>::iterator iter = this->cars.begin(); iter != this->cars.end(); iter ++) {
        if (iter->second.get_whether_preset() == 1) {
            this->preset_car_to_situation(iter->second);
            car new_car(iter->second);
            preset_car.push_back(new_car);
            preset_car_schedule_start_time[iter->first] = iter->second.get_schedule_start_time();
        }
    }
    this->arrive_car_id_count.clear();
    this->initial_cars_state_in_T0();
    this->schedule_cars(0);
    for (vector<car>::iterator iter = preset_car.begin(); iter != preset_car.end(); iter ++) {
        iter->set_schedule_start_time(-1 * (this->arrive_car_id_count[iter->get_id()] - iter->get_schedule_start_time() - this->estimate_spend_time(*iter)));
    }
    sort(preset_car.begin(), preset_car.end());
    int modify_n = preset_car.size() / 10;
    int epoch_modify_n = modify_n / this->my_config.epoch_to_modify_preset + 1;
    int doing_modify_n = 0;
    vector<car> save_car;
    for (vector<car>::iterator iter = preset_car.begin(); iter != preset_car.end(); iter ++) {
        cout << modify_n << endl;
        if (modify_n > 0) {
            int car_id = iter->get_id();
            car saving_car = this->cars[car_id];
            this->car_can_from_to = vector<map<int, map<int, int>>>(1, map<int, map<int, int>>());
            this->car_can_from_to_flag = vector<map<int, int>>(1, map<int, int>());
            this->car_can_from_to_time = 0;
            int flag = this->car_path_regulation(*iter, preset_car_schedule_start_time[car_id]);
            if (flag != 1) {
                this->cars[car_id] = saving_car;
                continue;
            } else {
                save_car.push_back(saving_car);
                doing_modify_n ++;
            }
        } else {
            break;
        }
        if (doing_modify_n >= epoch_modify_n || modify_n <= doing_modify_n) {
            this->arrive_car_id_count.clear();
            this->initial_cars_state_in_T0();
            int flag = this->schedule_cars(0);
            if (flag == -1) {
                for (vector<car>::iterator save_car_iter = save_car.begin(); save_car_iter != save_car.end(); save_car_iter ++)
                    this->cars[save_car_iter->get_id()] = *save_car_iter;
                save_car.clear();
            } else {
                modify_n -= doing_modify_n;
                doing_modify_n = 0;
                for (vector<car>::iterator save_car_iter = save_car.begin(); save_car_iter != save_car.end(); save_car_iter ++)
                    this->cars[save_car_iter->get_id()].set_whether_finish_find_path(2);
                save_car.clear();
            }
        }
    }
}

// regulate all cars start time and path
void overall_schedule::cars_path_regulation() {
    this->modify_preset_car();
    int rand_int = 1;
    // init situation car running in_road
    for (list<road>::iterator iter = this->roads_connect_cross.begin(); iter != this->roads_connect_cross.end(); iter ++) {
        iter->init_situation_car_running_in_road();
    }
    // Adjust the order of the vehicle from low to high speed anc count preset car to situation_car_running_in_road
    this->cars_to_regulate_path = vector<list<car>>(this->my_config.priority_N, list<car>());
    for (map<int, car>::iterator iter = this->cars.begin(); iter != this->cars.end(); iter ++) {
        if (iter->second.get_whether_preset() == 0 && iter->second.get_whether_finish_find_path() == 0) {
            car new_car(iter->second);
            //new_car.set_schedule_start_time(-1 * this->estimate_spend_time(new_car));
            new_car.set_schedule_start_time(new_car.get_plan_time());
            //new_car.set_schedule_start_time(-1 * new_car.get_speed());
            this->cars_to_regulate_path[new_car.get_priority()].push_back(new_car);
        } else {
            this->preset_car_to_situation(iter->second);
        }
    }
    for (int car_priority = this->my_config.priority_N - 1; car_priority >= 0; car_priority --) {
        // initial state
        this->arrive_car_id_count.clear();
        this->initial_cars_state_in_T0();

        int start_time = 1;
        int schedule_step = 1;
        this->cars_to_regulate_path[car_priority].sort();
        this->cars_arrive_schedule_start_time.clear();
        while (true) {
            int wait_schedule_time_car_N = 0;
            wait_schedule_time_car_N += this->cars_to_regulate_path[car_priority].size();
            int wait_run_car_N = cars_arrive_schedule_start_time.size();
            cout << "start_time = " << start_time << " remain_car_N = " << wait_schedule_time_car_N << " " << wait_run_car_N << endl;
            if (wait_schedule_time_car_N + wait_run_car_N == 0)
                break;
            while (!this->cars_to_regulate_path[car_priority].empty() && this->cars_to_regulate_path[car_priority].front().get_schedule_start_time() <= start_time) {
                car new_car(this->cars_to_regulate_path[car_priority].front());
                this->cars_to_regulate_path[car_priority].pop_front();
                new_car.set_schedule_start_time(-1 * this->estimate_spend_time(new_car));
                this->cars_arrive_schedule_start_time.push_back(new_car);
            }
            cout << "start sort" << endl;
            sort(this->cars_arrive_schedule_start_time.begin(), this->cars_arrive_schedule_start_time.end());
            cout << "finish sort" << endl;
            this->cars_start_run.clear();
            this->cars_wait_run.clear();
            int path_regulation_fail_count = 0;
            int path_regulation_success_count = 0;
            this->car_can_from_to = vector<map<int, map<int, int>>>(schedule_step, map<int, map<int, int>>());
            this->car_can_from_to_flag = vector<map<int, int>>(schedule_step, map<int, int>());
            for (vector<car>::iterator car_iter = this->cars_arrive_schedule_start_time.begin(); car_iter != this->cars_arrive_schedule_start_time.end(); car_iter ++) {
                int flag = -1;
                if (path_regulation_fail_count < this->my_config.max_path_regulation_fail_count && path_regulation_success_count < this->last_start_run_num)
                    for (int i = start_time; i < start_time + schedule_step; i += max(1, schedule_step / 10)) {
                        this->car_can_from_to_time = i - start_time;
                        if (this->car_can_from_to[this->car_can_from_to_time][car_iter->get_from()][car_iter->get_to()] == this->car_can_from_to_flag[this->car_can_from_to_time][car_iter->get_from()]) {
                            flag = this->car_path_regulation(*car_iter, start_time);
                            if (flag != -1)
                                break;
                        }
                        this->car_can_from_to_flag[this->car_can_from_to_time][car_iter->get_from()] ++;
                    }
                //cout << start_time << " " << flag << endl;
                if (flag != -1) {
                    // car start run
                    this->cars_start_run.push_back(*car_iter);
                    path_regulation_success_count ++;
                } else {
                    // car wait run
                    this->cars_wait_run.push_back(*car_iter);
                    path_regulation_fail_count ++;
                }
            }
            cout << "before " << this->cars_start_run.size() << " " << this->cars_wait_run.size() << endl;
            sort(this->cars_start_run.begin(), this->cars_start_run.end());
            this->save_overall_schedule_status();
            while (true) {
                this->load_overall_schedule_status();
                int count = 0;
                for (vector<car>::iterator iter = this->cars_start_run.begin(); iter != this->cars_start_run.end(); iter ++) {
                    if (this->cars[iter->get_id()].get_whether_finish_find_path() > 0) {
                        count ++;
                        this->crosses[iter->get_from()].add_car_wait_to_run_in_road(this->cars[iter->get_id()]);
                        this->cars_wait_schedule_start_time_n[iter->get_priority()] ++;
                    }
                }
                cout << "count = " << count << endl;
                if (this->schedule_cars(car_priority) > 0) {
                    int add_start_run_num = count - last_start_run_num;
                    this->last_start_run_num = count + max(add_start_run_num, this->my_config.max_add_start_run_num);
                    break;
                }
            }
            // update status to backup=================
            this->load_overall_schedule_status();
            for (vector<car>::iterator iter = this->cars_start_run.begin(); iter != this->cars_start_run.end(); iter ++) {
                if (this->cars[iter->get_id()].get_whether_finish_find_path() > 0) {
                    this->crosses[iter->get_from()].add_car_wait_to_run_in_road(this->cars[iter->get_id()]);
                    this->cars_wait_schedule_start_time_n[iter->get_priority()] ++;
                }
            }
            for (int i = 0; i < schedule_step; i ++)
                this->schedule_cars_one_time_unit(car_priority);
            // update status to backup=================
            cout << "after " << this->cars_start_run.size() << " " << this->cars_wait_run.size() << endl;
            for (vector<car>::iterator car_iter = this->cars_start_run.begin(); car_iter != this->cars_start_run.end(); car_iter ++) {
                if (this->cars[car_iter->get_id()].get_whether_finish_find_path() == 0) {
                    car_iter->set_schedule_start_time(-1 * this->estimate_spend_time(*car_iter));
                    this->cars_wait_run.push_back(*car_iter);
                } else {
                    this->cars[car_iter->get_id()].set_whether_finish_find_path(2);
                }
            }
            this->cars_arrive_schedule_start_time = this->cars_wait_run;
            start_time += schedule_step;
            if ((this->cars_arrive_schedule_start_time.size() + wait_schedule_time_car_N) < 5000)
                schedule_step = 1;
            else if (start_time > 100)
                schedule_step = 50;
            else if (this->cars_arrive_schedule_start_time.size() > 5000)
                schedule_step = 10;
            else if (start_time > 10)
                schedule_step = 5;
        }
    }
}

// regulate all cars start time and path
void overall_schedule::cars_together_path_regulation() {
    int rand_int = 1;
    // init situation car running in_road
    for (list<road>::iterator iter = this->roads_connect_cross.begin(); iter != this->roads_connect_cross.end(); iter ++) {
        iter->init_situation_car_running_in_road();
    }
    // Adjust the order of the vehicle from low to high speed anc count preset car to situation_car_running_in_road
    this->cars_to_regulate_path = vector<list<car>>(this->my_config.priority_N, list<car>());
    for (map<int, car>::iterator iter = this->cars.begin(); iter != this->cars.end(); iter ++) {
        if (iter->second.get_whether_preset() == 0) {
            car new_car(iter->second);
            //new_car.set_schedule_start_time(-1 * this->estimate_spend_time(new_car));
            new_car.set_schedule_start_time(new_car.get_plan_time());
            this->cars_to_regulate_path[new_car.get_priority()].push_back(new_car);
        } else {
            this->preset_car_to_situation(iter->second);
        }
    }
    // initial state
    this->arrive_car_id_count.clear();
    this->initial_cars_state_in_T0();

    int start_time = 1;
    int schedule_step = 1;
    for (int car_priority = this->my_config.priority_N - 1; car_priority >= 0; car_priority --)
        this->cars_to_regulate_path[car_priority].sort();
    this->cars_arrive_schedule_start_time.clear();
    while (true) {
        int wait_schedule_time_car_N = 0;
        for (int car_priority = this->my_config.priority_N - 1; car_priority >= 0; car_priority --)
            wait_schedule_time_car_N += this->cars_to_regulate_path[car_priority].size();
        int wait_run_car_N = cars_arrive_schedule_start_time.size();
        cout << "start_time = " << start_time << " remain_car_N = " << wait_schedule_time_car_N << " " << wait_run_car_N << endl;
        if (wait_schedule_time_car_N + wait_run_car_N == 0)
            break;
        for (int car_priority = this->my_config.priority_N - 1; car_priority >= 0; car_priority --)
            while (!this->cars_to_regulate_path[car_priority].empty() && this->cars_to_regulate_path[car_priority].front().get_schedule_start_time() <= start_time) {
                car new_car(this->cars_to_regulate_path[car_priority].front());
                this->cars_to_regulate_path[car_priority].pop_front();
                new_car.set_schedule_start_time(-1 * this->estimate_spend_time(new_car));
                this->cars_arrive_schedule_start_time.push_back(new_car);
            }
        cout << "start sort" << endl;
        sort(this->cars_arrive_schedule_start_time.begin(), this->cars_arrive_schedule_start_time.end());
        cout << "finish sort" << endl;
        this->cars_start_run.clear();
        this->cars_wait_run.clear();
        int path_regulation_fail_count = 0;
        this->car_can_from_to = vector<map<int, map<int, int>>>(schedule_step, map<int, map<int, int>>());
        this->car_can_from_to_flag = vector<map<int, int>>(schedule_step, map<int, int>());
        for (vector<car>::iterator car_iter = this->cars_arrive_schedule_start_time.begin(); car_iter != this->cars_arrive_schedule_start_time.end(); car_iter ++) {
            int flag = -1;
            if (path_regulation_fail_count < this->my_config.max_path_regulation_fail_count)
                for (int i = start_time; i < start_time + schedule_step; i += max(1, schedule_step / 10)) {
                    this->car_can_from_to_time = i - start_time;
                    if (this->car_can_from_to[this->car_can_from_to_time][car_iter->get_from()][car_iter->get_to()] == this->car_can_from_to_flag[this->car_can_from_to_time][car_iter->get_from()]) {
                        flag = this->car_path_regulation(*car_iter, start_time);
                        if (flag != -1)
                            break;
                    }
                    this->car_can_from_to_flag[this->car_can_from_to_time][car_iter->get_from()] ++;
                }
            //cout << start_time << " " << flag << endl;
            if (flag != -1) {
                // car start run
                this->cars_start_run.push_back(*car_iter);
            } else {
                // car wait run
                this->cars_wait_run.push_back(*car_iter);
                path_regulation_fail_count ++;
            }
        }
        cout << "before " << this->cars_start_run.size() << " " << this->cars_wait_run.size() << endl;
        sort(this->cars_start_run.begin(), this->cars_start_run.end());
        this->save_overall_schedule_status();
        while (true) {
            this->load_overall_schedule_status();
            int count = 0;
            for (vector<car>::iterator iter = this->cars_start_run.begin(); iter != this->cars_start_run.end(); iter ++) {
                if (this->cars[iter->get_id()].get_whether_finish_find_path() > 0) {
                    count ++;
                    this->crosses[iter->get_from()].add_car_wait_to_run_in_road(this->cars[iter->get_id()]);
                    this->cars_wait_schedule_start_time_n[iter->get_priority()] ++;
                }
            }
            cout << "count = " << count << endl;
            if (this->schedule_cars(0) > 0)
                break;
        }
        // update status to backup=================
        this->load_overall_schedule_status();
        for (vector<car>::iterator iter = this->cars_start_run.begin(); iter != this->cars_start_run.end(); iter ++) {
            if (this->cars[iter->get_id()].get_whether_finish_find_path() > 0) {
                this->crosses[iter->get_from()].add_car_wait_to_run_in_road(this->cars[iter->get_id()]);
                this->cars_wait_schedule_start_time_n[iter->get_priority()] ++;
            }
        }
        for (int i = 0; i < schedule_step; i ++)
            this->schedule_cars_one_time_unit(0);
        // update status to backup=================
        cout << "after " << this->cars_start_run.size() << " " << this->cars_wait_run.size() << endl;
        for (vector<car>::iterator car_iter = this->cars_start_run.begin(); car_iter != this->cars_start_run.end(); car_iter ++) {
            if (this->cars[car_iter->get_id()].get_whether_finish_find_path() == 0) {
                car_iter->set_schedule_start_time(-1 * this->estimate_spend_time(*car_iter));
                this->cars_wait_run.push_back(*car_iter);
            } else {
                this->cars[car_iter->get_id()].set_whether_finish_find_path(2);
            }
        }
        this->cars_arrive_schedule_start_time = this->cars_wait_run;
        start_time += schedule_step;
        if ((this->cars_arrive_schedule_start_time.size() + wait_schedule_time_car_N) < 5000)
            schedule_step = 1;
        else if (this->cars_arrive_schedule_start_time.size() > 5000)
            schedule_step = 50;
        else if (start_time > 100)
            schedule_step = 50;
        else if (start_time > 10)
            schedule_step = 5;
    }
}

// write answer to answer file
void overall_schedule::save_answer(string answer_path) {
    ofstream answer_info_file(answer_path);
    for (map<int, car>::iterator car_iter = this->cars.begin(); car_iter != this->cars.end(); car_iter ++) {
        if (car_iter->second.get_whether_preset() == 0 || car_iter->second.get_whether_finish_find_path() != 0)
            answer_info_file << car_iter->second.to_string() << endl;
    }
    answer_info_file.close();
}

// ==================backup status===================================
void overall_schedule::save_overall_schedule_status() {
    this->backup_arrive_car_id_count = this->arrive_car_id_count;
    this->backup_T = this->T;
    this->backup_arrive_T = this->arrive_T;
    this->backup_all_cars_running_time = this->all_cars_running_time;
    this->backup_cars_wait_schedule_start_time_n = this->cars_wait_schedule_start_time_n;
    this->backup_cars_running_n = this->cars_running_n; 
    this->backup_cars_arrive_destination_n = this->cars_arrive_destination_n;
    this->backup_cars_running_wait_state_n = this->cars_running_wait_state_n; 
    this->backup_cars_running_termination_state_n = this->cars_running_termination_state_n;
    this->backup_roads_connect_cross = this->roads_connect_cross;
}

void overall_schedule::load_overall_schedule_status() {
    this->arrive_car_id_count = this->backup_arrive_car_id_count;
    this->T = this->backup_T;
    this->arrive_T = this->backup_arrive_T;
    this->all_cars_running_time = this->backup_all_cars_running_time;
    this->cars_wait_schedule_start_time_n = this->backup_cars_wait_schedule_start_time_n;
    this->cars_running_n = this->backup_cars_running_n; 
    this->cars_arrive_destination_n = this->backup_cars_arrive_destination_n;
    this->cars_running_wait_state_n = this->backup_cars_running_wait_state_n; 
    this->cars_running_termination_state_n = this->backup_cars_running_termination_state_n;
    this->roads_connect_cross = this->backup_roads_connect_cross;
}
// ==================backup status===================================