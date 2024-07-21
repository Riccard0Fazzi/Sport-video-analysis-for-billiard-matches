#ifndef TRAJECTORY_TRACKING_H
#define TRAJECTORY_TRACKING_H

// Include necessary headers for OpenCV and other dependencies
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include "ball_detection.hpp"

// Declare function prototypes

/**
 * Track the balls given the first video frame with the bounding boxes of all balls
 * 
 * 
 * @param inputImage The input image in which the field is to be detected.
 * @return a vector of frames with the bounding boxes.
 */

void tracking_balls(std::vector<cv::Mat>& all_video_frames, std::vector<billiardBall>& balls, cv::Mat& H); 

#endif // TRAJECTORY_TRACKING_H

