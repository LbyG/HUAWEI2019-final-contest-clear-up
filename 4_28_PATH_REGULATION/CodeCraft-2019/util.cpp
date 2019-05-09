#include "util.h"
#include "cmp_car_id.h"
#include "cmp_car_schedule_start_time.h"
#include "cmp_car_dis_to_cross_and_channel_id.h"

#include <string>
#include <vector>
#include <queue>

using namespace std;

// split string to int vector by chars
vector<int> parse_string_to_int_vector(string info) {
    int len = info.size();
    vector<int> info_val;
    info_val.clear();
    int previous_char_pos = -1;
    int val = 0;
    int flag = 1;
    for (int i = 0; i < len; ++i) {
        if (info[i] < '0' || info[i] > '9') {
            if (i > (previous_char_pos + 1)) {
                info_val.push_back(flag * val);
                val = 0;
                flag = 1;
            }
            previous_char_pos = i;
            if (info[i] == '-')
                flag = -1;
            else
                flag = 1;
        } else {
            val = val * 10 + (int(info[i]) - int('0'));
        }
    }
    return info_val;
}

// function to clear priority queue
void clear_priority_queue(priority_queue<car, vector<car>, cmp_car_id> &pq) {
    while (! pq.empty()) {
        pq.pop();
    }
}

// function to clear priority queue
void clear_priority_queue(priority_queue<car, vector<car>, cmp_car_schedule_start_time> &pq) {
    while (! pq.empty()) {
        pq.pop();
    }
}

// function to clear priority queue
void clear_priority_queue(priority_queue<car, vector<car>, cmp_car_dis_to_cross_and_channel_id> &pq) {
    while (! pq.empty()) {
        pq.pop();
    }
}

// check whether all value in vector is 0, if all value is 0 return ture else return false
bool whether_vector_zero(vector<int> v) {
    for (vector<int>::iterator iter = v.begin(); iter != v.end(); iter ++)
        if (*iter > 0)
            return false;
    return true;
}