#ifndef FIELD_DETECTION_H
#define FIELD_DETECTION_H

// Include necessary headers for OpenCV and other dependencies
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Declare function prototypes

/**
 * Detects the field in the input image and returns a mask image
 * where the field region is marked.
 * 
 * @param inputImage The input image in which the field is to be detected.
 * @return A binary mask image where the detected field region is marked.
 */
cv::Mat field_detection(const cv::Mat& inputImage);

#endif // FIELD_DETECTION_H

