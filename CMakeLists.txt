cmake_minimum_required (VERSION 2.8)
project (dnntool)

file ( GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp )
file ( GLOB HEADERS ${PROJECT_SOURCE_DIR}/src/*.hpp )

find_package ( Boost COMPONENTS system filesystem program_options iostreams REQUIRED )
find_package ( OpenCV REQUIRED )
find_package ( yaml-cpp )

set ( DarkeNet_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/darknet )
set ( DarkNet_LIBS ${DarkeNet_DIR}/libdarknet.so )

add_definitions(-DOPENCV)

add_custom_target (
    darknet
    COMMAND make
    WORKING_DIRECTORY ${DarkeNet_DIR}
)

include_directories (
    src
    3rdparty/darknet/include
    3rdparty/darknet/src
)

set (
    LIBS
    ${DarkNet_LIBS}
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${Boost_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
)

find_package ( CUDA REQUIRED )

message(STATUS "CUDA Version: ${CUDA_VERSION_STRINGS}")
message(STATUS "CUDA Libararies: ${CUDA_LIBRARIES}")

set(
  CUDA_NVCC_FLAGS
  ${CUDA_NVCC_FLAGS};
  -O3
  -gencode arch=compute_30,code=sm_30
  -gencode arch=compute_35,code=sm_35
  -gencode arch=compute_50,code=[sm_50,compute_50]
  -gencode arch=compute_52,code=[sm_52,compute_52]
)

add_definitions(-DGPU)
add_definitions(-D_FORCE_INLINES)
add_definitions(-O4 -g)

set (
    LIBS
    ${LIBS}
    ${CUDA_LIBRARIES}
)

cuda_add_executable (
    dnntool
    ${SOURCES}
)

add_dependencies(
    dnntool
    darknet
)

target_link_libraries (
    dnntool
    ${LIBS}
    ${CUDA_LIBRARIES}
)
