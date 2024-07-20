#ifndef VISUALIZATION
#define VISUALIZATION

#include <opencv2/opencv.hpp>
#include <iostream>

struct colors{
    cv::Scalar white = cv::Scalar (255,255,255);
    cv::Scalar black = cv::Scalar (64,64,64);
    cv::Scalar solid = cv::Scalar (255,153,255);
    cv::Scalar stripes = cv::Scalar (153,255,204);
};

void drawBallsOnTopView(const std::vector<billiardBall>& balls, cv::Mat& top_view);

#endif // VISUALIZATION