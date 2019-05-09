#include "config.h"

#include <algorithm>

using namespace std;

config::config() {
    this->channel_weight = vector<double>(this->max_channel + 1);
    this->channel_weight[1] = 1;
    for (int i = 2; i <= this->max_channel; i ++) {
        this->channel_weight[i] = this->channel_weight[i - 1] * this->channel_metric;//0.5;
    }
    for (int i = 2; i <= this->max_channel; i ++) {
        this->channel_weight[i] += this->channel_weight[i - 1];
    }
}

double config::count_capacity(int channel, int length, int speed) {
    if (channel < this->max_channel)
        return this->capacity_metric * this->channel_weight[channel] * max(1, length - speed);
    else {
        this->max_channel = channel;
        this->channel_weight = vector<double>(this->max_channel + 1);
        this->channel_weight[1] = 1;
        for (int i = 2; i <= this->max_channel; i ++) {
            this->channel_weight[i] = this->channel_weight[i - 1] * this->channel_metric;//0.5;
        }
        for (int i = 2; i <= this->max_channel; i ++) {
            this->channel_weight[i] += this->channel_weight[i - 1];
        }
    }
}