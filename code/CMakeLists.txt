cmake_minimum_required(VERSION 3.17)

set(PROJECT_NAME MCGP)
set(CODE_PATH src)

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")
set(EXECUTABLE_OUTPUT_PATH "../..")

project(${PROJECT_NAME})
include_directories(${CODE_PATH})
file(GLOB_RECURSE CODE_FILES ${CODE_PATH}/*.*)
add_executable(${PROJECT_NAME} ${CODE_FILES})
