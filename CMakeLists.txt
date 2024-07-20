cmake_minimum_required(VERSION 2.8)
project(Sport-video-analysis-for-billiard-matches)

# Specify OpenCV directory
set(OpenCV_DIR "/opt/homebrew/Cellar/opencv/4.9.0_11.reinstall/lib/cmake/opencv4")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "/opt/homebrew/opt/ffmpeg@6/lib;/opt/homebrew/Cellar/ffmpeg@6/lib")

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Source files
set(SOURCES src/main2.cpp)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    include  # Add your include directory where field_detection.h is located
)

# Add executable
add_executable(Sport-video-analysis-for-billiard-matches
    src/field_detection.cpp
	src/ball_detection.cpp
	src/homography.cpp
    src/main2.cpp
    #src/classifier.cpp
    # Include other source files if needed (ball_classification.cpp, tracking.cpp, etc.)
)

# Link OpenCV libraries
target_link_libraries(Sport-video-analysis-for-billiard-matches ${OpenCV_LIBS})

