//
// Created by jose on 5/04/23.
//
#ifndef MLFORECAST_ROLLING_H
#define MLFORECAST_ROLLING_H
#endif //MLFORECAST_ROLLING_H

class RollingMean {
private:
    int window_size_;
    int min_samples_;
public:
    RollingMean(int window_size, int min_samples) : window_size_(window_size), min_samples_(min_samples) {}
    void transform(float* data, int32_t n_data, float* out);
    float update(float* data, int32_t n_data);
};
