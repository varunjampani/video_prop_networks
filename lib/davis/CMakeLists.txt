# ----------------------------------------------------------------------------
# A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2016 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 2.8)
project( davis )

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

add_definitions(-std=c++11)
if(APPLE)
	add_definitions(-Wno-return-type)
endif()

# Finding python and boost-python
find_package(NumPy REQUIRED)
find_package(Boost COMPONENTS python)
find_package(PythonInterp 2.7.6 REQUIRED)
find_package(PythonLibs   2.7.6 REQUIRED)

include_directories( ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS}
	${NUMPY_INCLUDE_DIRS} cpp cpp/Boost.NumPy cpp/polycont)

link_directories( ${Boost_LIBRARY_DIR} )

add_subdirectory(cpp)

# Create python library
add_library(tstab SHARED cpp/polycont/python.cpp cpp/clipper/clipper.cpp)

target_link_libraries( tstab boost_numpy ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} )
set_target_properties( tstab PROPERTIES PREFIX "")

if(APPLE)
	set_target_properties( tstab PROPERTIES SUFFIX ".so" )
endif()
