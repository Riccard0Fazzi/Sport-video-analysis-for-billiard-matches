#ifndef SPORT_VIDEO_ANALYSIS_FOR_BILLIARD_MATCHES_CLASSIFICATION_H
#define SPORT_VIDEO_ANALYSIS_FOR_BILLIARD_MATCHES_CLASSIFICATION_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "ball_detection.hpp"

void classification(const cv::Mat& image, std::vector<billiardBall>& balls);

#endif //SPORT_VIDEO_ANALYSIS_FOR_BILLIARD_MATCHES_CLASSIFICATION_H
