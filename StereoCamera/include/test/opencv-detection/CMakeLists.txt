cmake_minimum_required(VERSION 2.8)
project( detection )
find_package( OpenCV REQUIRED )
add_executable( detection detection.cpp )
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb ")
target_link_libraries( detection ${OpenCV_LIBS} )
target_link_libraries( detection
	/home/linux/FaceAlignment/test/opencv-detection/libfacealignment.so)
