//
// Created by jose on 5/04/23.
//
#include <limits>
#include <cstdint>

#include "rolling.h"

void RollingMean::transform(float *data, int32_t n_data, float *out) {
    float accum = 0.0;
    for (int i = 0; i < min_samples_ - 1; ++i) {
        accum += data[i];
        out[i] = std::numeric_limits<float>::quiet_NaN();
    }
    for (int i = min_samples_ - 1; i < window_size_; ++i) {
        accum += data[i];
        out[i] = accum / (i + 1);
    }
    for (int i = window_size_; i < n_data; ++i) {
        accum += data[i] - data[i - window_size_];
        out[i] = accum / window_size_;
    }
}

float RollingMean::update(float *data, int32_t n_data) {
    float accum = 0.0;
    if (n_data < min_samples_) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    int n_samples = window_size_ > n_data ? n_data : window_size_;
    for (int i = 0; i < n_samples; ++i) {
        accum += data[n_data - 1 - i];
    }
    return accum / n_samples;
}
