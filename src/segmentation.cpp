#include "../include/segmentation.h"

void segmentation(const cv::Mat& image, const std::vector<cv::Point>& field_points, const std::vector<billiardBall>& balls, cv::Mat& segmented_image) {
    // Copy the original image to the segmented one in order to apply the table and balls shapes
    image.copyTo(segmented_image);

    // Define the color palette
    colors colors;
    colors.white = cv::Scalar (255,255,255);
    colors.black = cv::Scalar (0,0,0);
    colors.solid = cv::Scalar (255,153,51);
    colors.stripes = cv::Scalar (51,51,255);
    colors.playing_field = cv::Scalar (0,153,0);
    colors.field_border = cv::Scalar (51,255,255);

    // Draw the lines for the table border
    cv::line(segmented_image,field_points[0],field_points[1],colors.field_border);
    cv::line(segmented_image,field_points[1],field_points[2],colors.field_border);
    cv::line(segmented_image,field_points[2],field_points[3],colors.field_border);
    cv::line(segmented_image,field_points[3],field_points[0],colors.field_border);

    // Draw the table area
    cv::fillConvexPoly(segmented_image,field_points,colors.playing_field);

    // Draw the balls
    for (const billiardBall& ball: balls) {
        if (ball.id == 0)
            cv::circle(segmented_image,cv::Point(ball.x,ball.y),ball.true_radius,colors.white,-1);
        else if (ball.id == 1)
            cv::circle(segmented_image,cv::Point(ball.x,ball.y),ball.true_radius,colors.black,-1);
        else if (ball.id == 2)
            cv::circle(segmented_image,cv::Point(ball.x,ball.y),ball.true_radius,colors.solid,-1);
        else
            cv::circle(segmented_image,cv::Point(ball.x,ball.y),ball.true_radius,colors.stripes,-1);
    }
}

void performanceSegmentation(const cv::Mat& image, const std::vector<cv::Point>& field_points, const std::vector<billiardBall>& balls, cv::Mat& segmented_image) {
    segmented_image = cv::Mat::zeros(image.size(), CV_8U);

    // Define the color palette
    colors colors;
    colors.white = cv::Scalar (1);
    colors.black = cv::Scalar (2);
    colors.solid = cv::Scalar (3);
    colors.stripes = cv::Scalar (4);
    colors.playing_field = cv::Scalar (5);

    // Draw the table area
    cv::fillConvexPoly(segmented_image,field_points,colors.playing_field);

    // Draw the balls
    for (const billiardBall& ball: balls) {
        if (ball.id == 0)
            cv::circle(segmented_image,cv::Point(ball.x,ball.y),ball.true_radius,colors.white,-1);
        else if (ball.id == 1)
            cv::circle(segmented_image,cv::Point(ball.x,ball.y),ball.true_radius,colors.black,-1);
        else if (ball.id == 2)
            cv::circle(segmented_image,cv::Point(ball.x,ball.y),ball.true_radius,colors.solid,-1);
        else
            cv::circle(segmented_image,cv::Point(ball.x,ball.y),ball.true_radius,colors.stripes,-1);
    }
}
