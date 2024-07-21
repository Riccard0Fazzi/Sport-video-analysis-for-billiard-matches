#ifndef HOMOGRAPHY
#define HOMOGRAPHY

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include "ball_detection.h"

// Given a set of 4 points, it computes the homography transformation associated
cv::Mat computeHomography(const std::vector<cv::Point>& points);

// Applies the transformaation H on a set of points
void mapPoints(const cv::Mat H, const std::vector<cv::Point>& input_points, std::vector<cv::Point>& output_points);

// Given the corners of the billiard table 'field_points' and the billiard balls positions 'balls_coords', applies the
// perspective transformation on the balls coordinates
void applyPerspectiveTransform(const std::vector<cv::Point>& field_points, const std::vector<cv::Point>& balls_coords,
                               std::vector<cv::Point>& transformed_balls_coords);

#endif // HOMOGRAPHY

