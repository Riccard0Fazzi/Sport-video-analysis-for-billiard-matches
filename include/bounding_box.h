#ifndef SPORT_VIDEO_ANALYSIS_FOR_BILLIARD_MATCHES_BOUNDING_BOX_H
#define SPORT_VIDEO_ANALYSIS_FOR_BILLIARD_MATCHES_BOUNDING_BOX_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "ball_detection.hpp"

struct ballBoundingBox {
    cv::Rect bbox;
    int id;

    ballBoundingBox(cv::Rect& bbox, int id);

    ~ballBoundingBox() {}
};

struct colors {
    cv::Scalar white;
    cv::Scalar black;
    cv::Scalar solid;
    cv::Scalar stripes;
    cv::Scalar playing_field;
    cv::Scalar background;
    cv::Scalar field_border;
};

void createBoundingBoxes(const std::vector<billiardBall>& balls, std::vector<ballBoundingBox>& bboxes);

#endif //SPORT_VIDEO_ANALYSIS_FOR_BILLIARD_MATCHES_BOUNDING_BOX_H
