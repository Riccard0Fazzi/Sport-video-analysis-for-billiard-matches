cmake_minimum_required(VERSION 2.8)
project(MyProject)

# Include directory for header files
include_directories(include)

# Find OpenCV package
find_package(OpenCV 4 REQUIRED)

# Add executable
add_executable(my_program src/ImageStitcher.cpp src/main.cpp)

# Link against OpenCV libraries
target_link_libraries(my_program ${OpenCV_LIBS})

