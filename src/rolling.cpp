//
// Created by jose on 5/04/23.
//
#include <limits>
#include <cstdint>

#include "rolling.h"

void my_rolling_mean(float* data, int32_t n_data, int32_t window_size, int32_t min_samples, float* out) {
    float accum = 0.0;
    for (int i = 0; i < min_samples - 1; ++i) {
        accum += data[i];
        out[i] = std::numeric_limits<float>::quiet_NaN();
    }
    for (int i = min_samples - 1; i < window_size; ++i) {
        accum += data[i];
        out[i] = accum / (i + 1);
    }
    for (int i = window_size; i < n_data; ++i) {
        accum += data[i] - data[i - window_size];
        out[i] = accum / window_size;
    }
}
