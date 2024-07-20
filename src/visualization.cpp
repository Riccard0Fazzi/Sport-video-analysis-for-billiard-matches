#include "ball_detection.hpp"
#include "visualization.hpp"

void drawBallsOnTopView(const std::vector<billiardBall>& balls, cv::Mat& top_view) {
    // Read the top-view image from the data folder
    top_view = imread("../data/Top_View.jpg", cv::IMREAD_ANYCOLOR);

    // Initialize the colors
    colors colors;

    // Draw the points on the top view
    for (const billiardBall& ball : balls) {
        if (ball.ball_category_ID == 0)
            cv::circle(top_view,cv::Point (ball.x,ball.y),5,colors.white,-1);
        else if (ball.ball_category_ID == 1)
            cv::circle(top_view,cv::Point (ball.x,ball.y),5,colors.black,-1);
        else if (ball.ball_category_ID == 2)
            cv::circle(top_view,cv::Point (ball.x,ball.y),5,colors.solid,-1);
        else
            cv::circle(top_view,cv::Point (ball.x,ball.y),5,colors.stripes,-1);

        cv::circle(top_view,cv::Point (ball.x,ball.y),5,cv::Scalar(0,0,0),1);
    }
}