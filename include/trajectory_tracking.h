#ifndef TRAJECTORY_TRACKING_H
#define TRAJECTORY_TRACKING_H

// Include necessary headers for OpenCV and other dependencies
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>

// Declare function prototypes

/**
 * Track the balls given the first video frame with the bounding boxes of all balls
 * 
 * 
 * @param inputImage The input image in which the field is to be detected.
 * @return a vector of frames with the bounding boxes.
 */
std::vector<cv::Mat> detectField(const cv::Mat& inputImage);


#endif // TRAJECTORY_TRACKING_H

