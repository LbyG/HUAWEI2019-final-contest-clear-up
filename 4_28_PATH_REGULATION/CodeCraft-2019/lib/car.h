#ifndef CAR_H
#define CAR_H

#include <string>
#include <list>
#include <vector>

using namespace std;

class car {
private:
    // car id
    int id; 
    // start from which cross_id
    int from;
    // arrive to which cross_id;
    int to; 
    // car max speed limit
    int speed; 
    // car plan start time
    int plan_time; 
    // car priority
    int priority;
    // car whether preset
    int whether_preset;
    // car whether finish find path
    int whether_finish_find_path;
    
    // car schedule start time
    int schedule_start_time;
    // car schedule path
    list<int> schedule_path;
    
    // schedule_status == 0 termination state; schedule_status == 1 wait state;
    int schedule_status; 
    // distance from the car to the cross
    int dis_to_cross;
    // channel id which car running in
    int channel_id;
    
    // ==============================================================
    // path regulation
    
public:
    car();
    // car_info = (id,from,to,speed,planTime)
    car(string car_info); 
    // schedule_info = [id, schedule_start_time, schedule_path1, schedule_path2, schedule_path3, schedule_path4, ...]
    void set_schedule_path(vector<int> schedule_info);
    // return car id
    int get_id() const; 
    // reutrn car start from cross_id;
    int get_from() const; 
    // return car arrive to cross_id;
    int get_to() const;
    // return car speed
    int get_speed() const; 
    // return car plan start time
    int get_plan_time() const; 
    // return car priority
    int get_priority() const; 
    // return car whether preset
    int get_whether_preset() const;
    
    // set whether_finish_find_path
    void set_whether_finish_find_path(int whether_finish_find_path);
    // return whether_finish_find_path
    int get_whether_finish_find_path() const;
    
    void set_schedule_start_time(int time);
    // return car schedule start time
    int get_schedule_start_time() const;
    // return next road_id in path, if path is empty return -1
    int get_next_road_in_path() const;
    // arrive next road, so this->schedule_path.pop_front();
    void arrive_next_road_path();
    
    // set this -> schedule_status = schedule_status
    void set_schedule_status(int schedule_status);
    // return car schedule status
    int get_schedule_status() const; 
    // set this -> dis_to_cross = dis_to_cross
    void set_dis_to_cross(int dis_to_cross);
    // return distance from the car to the cross
    int get_dis_to_cross() const;
    // set this -> channel_id = channel_id
    void set_channel_id(int channel_id);
    // return channel id which car running in
    int get_channel_id() const;
    
    // ==============================================================
    // path regulation
    
    string to_string();
};

bool operator<(const car &a, const car &b);

#endif