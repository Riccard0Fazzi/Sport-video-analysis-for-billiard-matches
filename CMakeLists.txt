cmake_minimum_required(VERSION 2.8)
project(BiliardVideoAnalysis)

# Include directory for header files
include_directories(include)

# Find OpenCV package
find_package(OpenCV 4 REQUIRED)

# Add executable
add_executable(${PROJECT_NAME}
        src/circleDetector.cpp)

# Link against OpenCV libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})