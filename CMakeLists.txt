cmake_minimum_required (VERSION 2.8)
project (F_test)

set(CMAKE_CXX_STANDARD 11)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(PUBLIC
        ${EIGEN3_INCLUDE_DIR})

add_executable(main main.cc)

target_link_libraries(main)
