#ifndef HOMOGRAPHY
#define HOMOGRAPHY

// Include necessary headers for OpenCV and other dependencies
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
// Declare function prototypes

/**
 * Apply the homography to a set of points
	 */
void homography(const std::vector<cv::Point>& points);
#endif // HOMOGRAPHY

