#ifndef BALL_DETECTION_H
#define BALL_DETECTION_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/imgproc.hpp>


// Structure used to store balls names and colors
struct billiardBall {
	int x,y;
	int width, height;
    int ball_category_ID;
	cv::Mat image;
	std::string color_name;
    cv::Vec3b color_value;

	billiardBall(int x, int y, int width, int height, cv::Mat& image);
};


// Structure used to contain each billiard sets of balls of different colors
struct billiardSet {
    std::string table;
    std::vector<billiardBall> billiard_set;
};


// Computes the most common color in 'img' via histogram evaluation
// in HSV color-space. Every color is considered, except for black,
// i.e. (0,0,0).
// The most common color is given by the Vec3b argument.
void mostCommonColor(const cv::Mat& img, cv::Vec3b& most_common_color);


// Applies contrast stretching adaptively to 'img' in order to
// enhance the darker regions.
// The function works by first computing the standard deviation
// of the input image, and then by linearly changing the pixels
// values proportionally to the standard deviation. Everything
// is performed w.r.t. the V channel of the HSV color-space.
void contrastStretching(const cv::Mat& img, cv::Mat& dest);


// The function has the objective of removing the most common color on 'img'.
// This is done adaptively by computing the most common color of each sub-window
// of the image, where the size of the sub-windows is proportional to 'window_ratio'.
// By making use of the sub-windows we achieve much more invariance to illumination.
// Lastly, we apply the thresholding of the dominant color using the tolerances given
// by 'HSV_thresholds'.
void adaptiveColorBasedSegmentation(const cv::Mat& img, cv::Mat& dest, std::vector<int> HSV_thresholds, double window_ratio);


// Selects the detected balls based on their colors and on the color palettes
// given by the struct 'billiard_tables'.
// The selected colors are then returned by 'new_circle_vector'
void ballSelection(const cv::Mat& img, const std::vector<cv::Vec3f>& circle_vector,
                   std::vector<cv::Vec3f>& new_circle_vector, std::vector<billiardSet> billiard_tables);


// Overall procedure for detecting and localizing billiard balls on the game
// table.
// The following steps are pursued:  - Bilateral filtering
//                                   - Contrast stretching
//                                   - Adaptive Color-based Segmentation
//                                   - Mask binarization
//                                   - Morphological operators (Closing + Opening)
//                                   - Hough Circle Transform
//                                   - Ball Selection

void ballDetection(const cv::Mat& img, std::vector<cv::Vec3f>& circles);

std::vector<cv::Vec3f> ball_detection(std::vector<cv::Point> field_points, const cv::Mat& inputImage);

// Draws the circles present on the vector 'circles'.
void drawCircles(const cv::Mat& img, cv::Mat& circles_img, const std::vector<cv::Vec3f>& circles);


void printCircles(const cv::Mat& img, const std::vector<cv::Vec3f>& circles, int circles_img_size, std::vector<cv::Mat>& circles_img);

#endif //BALL_DETECTION_H
