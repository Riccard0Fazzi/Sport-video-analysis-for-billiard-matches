cmake_minimum_required(VERSION 2.8)
project(BiliardVideoAnalysis)

# Link against OpenCV libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Source files
set(SOURCES src/main.cpp)

# Add executable
add_executable(${PROJECT_NAME}
        src/ball_detection.cpp)

# Link OpenCV libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

