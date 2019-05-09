#ifndef UTIL_H
#define UTIL_H

#include "cmp_car_id.h"
#include "cmp_car_schedule_start_time.h"
#include "cmp_car_dis_to_cross_and_channel_id.h"

#include <string>
#include <vector>
#include <queue>

using namespace std;

// split string to int vector by chars
vector<int> parse_string_to_int_vector(string info);
// function to clear priority queue
void clear_priority_queue(priority_queue<car, vector<car>, cmp_car_id> &pq);
void clear_priority_queue(priority_queue<car, vector<car>, cmp_car_schedule_start_time> &pq);
void clear_priority_queue(priority_queue<car, vector<car>, cmp_car_dis_to_cross_and_channel_id> &pq);
// check whether all value in vector is 0, if all value is 0 return ture else return false
bool whether_vector_zero(vector<int> v);

#endif