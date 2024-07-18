#ifndef CLASSIFIER
#define CLASSIFIER

// Include necessary headers for OpenCV and other dependencies
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

void classify(std::vector<billiardBall> &balls, cv::Mat& table_image);
#endif // CLASSIFIER

