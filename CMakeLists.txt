cmake_minimum_required(VERSION 3.20)
project(ASC_ode)

set (CMAKE_CXX_STANDARD 20)


include_directories(src nanoblas/src)

add_subdirectory (src)
add_subdirectory (nanoblas)


add_executable (test_ode demos/test_ode.cpp)
target_link_libraries (test_ode PUBLIC nanoblas)


add_executable (test_ode_ad demos/test_ode_ad.cpp)
target_link_libraries (test_ode_ad PUBLIC nanoblas)


add_executable (test_ode_Electric demos/test_ode_Electric.cpp)
target_link_libraries (test_ode_Electric PUBLIC nanoblas)


add_executable (demo_autodiff demos/demo_autodiff.cpp)
target_link_libraries (demo_autodiff PUBLIC nanoblas)


add_executable (polynomials demos/polynomials.cpp)
target_link_libraries (polynomials PUBLIC nanoblas)
