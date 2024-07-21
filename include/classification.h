#ifndef SPORT_VIDEO_ANALYSIS_FOR_BILLIARD_MATCHES_CLASSIFICATION_H
#define SPORT_VIDEO_ANALYSIS_FOR_BILLIARD_MATCHES_CLASSIFICATION_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "ball_detection.h"

void printCircles(const cv::Mat& img, const std::vector<billiardBall>& balls, std::vector<cv::Mat>& circles_img);

void classification(const cv::Mat& img, std::vector<billiardBall>& balls);

#endif //SPORT_VIDEO_ANALYSIS_FOR_BILLIARD_MATCHES_CLASSIFICATION_H
