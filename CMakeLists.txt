cmake_minimum_required(VERSION 3.25)
project(mlforecast)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

add_library(mlforecast SHARED src/grouped_array.cpp src/rolling.h src/rolling.cpp src/boost_rolling_mean.h src/boost_rolling_mean.cpp)
