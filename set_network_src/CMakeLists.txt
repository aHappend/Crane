cmake_minimum_required(VERSION 3.16)
project(set_network_src CXX)

set(CMAKE_CXX_STANDARD 17)

file(GLOB NETWORK_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(set_network_profiles STATIC ${NETWORK_SRC})
target_include_directories(set_network_profiles PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
