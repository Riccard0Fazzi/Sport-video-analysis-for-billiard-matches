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

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    include  # Add your include directory where field_detection.h is located
)

# Add executable
add_executable(BiliardVideoAnalysis
    src/field_detection.cpp
    src/main.cpp
    # Include other source files if needed (ball_classification.cpp, tracking.cpp, etc.)
)

# Link OpenCV libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

