#ifndef SPORT_VIDEO_ANALYSIS_FOR_BILLIARD_MATCHES_SEGMENTATION_H
#define SPORT_VIDEO_ANALYSIS_FOR_BILLIARD_MATCHES_SEGMENTATION_H

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

void segmentation(const cv::Mat& image, const std::vector<cv::Point>& field_points, const std::vector<billiardBall>& balls, cv::Mat& segmented_image);

void performanceSegmentation(const cv::Mat& image, const std::vector<cv::Point>& field_points, const std::vector<billiardBall>& balls, cv::Mat& segmented_image);

#endif //SPORT_VIDEO_ANALYSIS_FOR_BILLIARD_MATCHES_SEGMENTATION_H
