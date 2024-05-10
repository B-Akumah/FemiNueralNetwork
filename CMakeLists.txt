cmake_minimum_required(VERSION 3.27)
project(FemiTest)

set(CMAKE_CXX_STANDARD 17)

add_executable(FemiTest main.cpp
        matrix.hpp
        nueral_network.hpp
        test_xor.cpp)
