//
// Created by mrjak on 21-04-2022.
//

#ifndef GPU_SYNC_TIMER_H
#define GPU_SYNC_TIMER_H

#include <ctime>
#include <iostream>
//#include <string>
#include <vector>
#include <map>
#include <ATen/ATen.h>
//#include <torch/extension.h>

template <unsigned int number_of_stages>
class Timer {
protected:
    //iterations
    std::clock_t itr_time_start;
    std::clock_t itr_time_end;
    std::vector<double> iterations;

    //stages
    double start_stage_times[number_of_stages];
    double end_stage_times[number_of_stages];
    double total_stage_times[number_of_stages];

public:
    Timer() {
        for(int i =0;i<number_of_stages;i++){
            total_stage_times[i] = 0.;
        }
    }

    void start_itr_time() {
        itr_time_start = std::clock();
    }

    void end_itr_time() {
        itr_time_end = std::clock();
        double duration = (itr_time_end - itr_time_start) / (double) (CLOCKS_PER_SEC);
        iterations.push_back(duration);
    }

    void start_stage_time(int s) {
        start_stage_times[s] = std::clock();
    }

    void end_stage_time(int s) {
        end_stage_times[s] = std::clock();
        double duration = (end_stage_times[s] - start_stage_times[s]) / (double) (CLOCKS_PER_SEC);
        total_stage_times[s] += duration;
    }

    at::Tensor get_itr_times() {
        int number_of_itrs = iterations.size();
        at::Tensor r = at::zeros(number_of_itrs);
        for (int i = 0; i < number_of_itrs; i++) {
            r[i] = iterations[i];
        }
        return r;
    }

    at::Tensor get_stage_times() {
//        int number_of_stages = total_stage_times.size();
        at::Tensor r = at::zeros(number_of_stages);
        for (int i = 0; i < number_of_stages; i++) {
            r[i] = total_stage_times[i];
        }
        return r;
    }
};


#endif //GPU_SYNC_TIMER_H
