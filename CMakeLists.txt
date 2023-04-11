cmake_minimum_required(VERSION 3.25)
project(mlforecast)

set(CMAKE_CXX_STANDARD 17)

add_library(mlforecast SHARED src/grouped_array.cpp src/rolling.h src/rolling.cpp)
