#ifndef OVERALL_SCHEDULE
#define OVERALL_SCHEDULE

#include <queue>
#include <map>
#include <list>

#include "config.h"
#include "car.h"
#include "road.h"
#include "cross.h"
#include "cmp_car_schedule_start_time.h"
#include "cmp_car_id.h"
#include "car_influence_to_road.h"

using namespace std;

class overall_schedule {
private:
    config my_config;
    int last_start_run_num = this->my_config.max_add_start_run_num;
    int deadlock_step = 0;
    int deadlock_epoch_step = 0;
    long double metric_a;
    long double metric_b;
    vector<long double> priority_weight;
    // arrive car id count
    map<int, int> arrive_car_id_count;
    // timestamp
    int T;
    // different priority cars all arrive destination time
    vector<int> arrive_T;
    // different priority sum of cars running time 
    vector<long long> all_cars_running_time;
    // N = cars_wait_schedule_start_time_n + cars_wait_run_n + cars_running_n + cars_arrive_destination_n
    // number of cars which T < car.schedule_start_time
    vector<int> cars_wait_schedule_start_time_n;
    // the number of cars which is running in road
    vector<int> cars_running_n; 
    // the number of cars which is arrive destination
    vector<int> cars_arrive_destination_n;
    
    // cars_running_n = cars_wait_state_n + cars_termination_state_n
    // the number of cars wait to through cross or previous car is wait to through cross
    int cars_running_wait_state_n; 
    // the number of cars was run in this time unit
    int cars_running_termination_state_n;
    
    // cars[car.id] = car
    map<int, car> cars; 
    vector<list<car>> cars_to_regulate_path;
    vector<car> cars_arrive_schedule_start_time;
    vector<car> cars_start_run;
    vector<car> cars_wait_run;
    //record car influence to road
    map<int, car_influence_to_road> cars_influence_to_road;
    
    // roads[road.id] = road
    map<int, road> roads; 
    // crosses[cross.id] = cross
    map<int, cross> crosses;
    // vector of roads connect to cross
    list<road> roads_connect_cross;
    
    // if car_can_from_to[time][from][to] == car_can_from_to_flag[time][from], then car can start from->to in time
    vector<map<int, map<int, int>>> car_can_from_to;
    vector<map<int, int>> car_can_from_to_flag;
    // car start time in path regulation
    int car_can_from_to_time;
    
    // ==================backup status===================================
    map<int, int> backup_arrive_car_id_count;
    int backup_T;
    vector<int> backup_arrive_T;
    vector<long long> backup_all_cars_running_time;
    vector<int> backup_cars_wait_schedule_start_time_n;
    vector<int> backup_cars_running_n; 
    vector<int> backup_cars_arrive_destination_n;
    int backup_cars_running_wait_state_n; 
    int backup_cars_running_termination_state_n;
    list<road> backup_roads_connect_cross;
    // ==================backup status===================================
public:
    // return cars number
    int get_cars_n();
    
    //load cars, roads and crosses info from car_path, road_path and cross_path file
    void load_cars_roads_crosses(string car_path, string road_path, string cross_path);
    // load cars' path from answer_path
    void load_answer(string answer_path);
    // load preset answer
    void load_preset_answer(string presetAnswerPath);
    
    // initial all data in T = 0
    void initial_cars_state_in_T0();
    // Schedule cars in road.
    // If car can through cross then car into schedule wait -> car.schedule_status = 1
    // If car blocked by schedule wait car then car into schedule wait -> car.schedule_status = 1
    // If car don't be block and can't through cross then car run one time slice and into end state -> car.schedule_status = 2
    // If car blocked by end state car then car move to the back of the previous car
    void schedule_cars_running_in_road();
    // Schedule cars which arrive schedule time or wait start
    // cars_wait_schedule_start_time_list -> cars_wait_run_list
    // cars_wait_run_list -> cars_in_road
    void schedule_cars_wait_run(int cars_wait_run_priority);
    
    // schedule cars in one time unit
    // if deadblock return false
    bool schedule_cars_one_time_unit(int car_priority);
    // return all cars arrive to_cross_id need how much times
    int schedule_cars(int car_priority); 
    
    // count metric a and metric b
    void count_metric_ab();
    
    // output car schedule status
    void output_schedule_status();
    
    //================================================================
    
    void modify_preset_car();
    
    // count preset car to situation_car_running_in_road
    void preset_car_to_situation(car preset_car);
    
    // some way to prevent deadlock
    void prevent_deadlock(int T, int car_priority);
    
    // estimate time car go to destination
    int estimate_spend_time(car estimate_car);
    // reluate car start time and path
    int car_path_regulation(car car_iter, int start_time);
    // regulate all cars start time and path
    void cars_path_regulation();
    void cars_together_path_regulation();
    
    // write answer to answer file
    void save_answer(string answer_path);
    
    // ==================backup status===================================
    void save_overall_schedule_status();
    void load_overall_schedule_status();
    // ==================backup status===================================
};

#endif