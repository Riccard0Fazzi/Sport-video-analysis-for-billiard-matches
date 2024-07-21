#ifndef SPORT_VIDEO_ANALYSIS_FOR_BILLIARD_MATCHES_BOUNDING_BOX_H
#define SPORT_VIDEO_ANALYSIS_FOR_BILLIARD_MATCHES_BOUNDING_BOX_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "ball_detection.h"


struct colors {
    cv::Scalar white;
    cv::Scalar black;
    cv::Scalar solid;
    cv::Scalar stripes;
    cv::Scalar playing_field;
    cv::Scalar background;
    cv::Scalar field_border;
};

void createBoundingBoxes(const std::vector<billiardBall>& balls, cv::Mat& field_image, cv::Mat& field_bounding_box); 
#endif //SPORT_VIDEO_ANALYSIS_FOR_BILLIARD_MATCHES_BOUNDING_BOX_H
