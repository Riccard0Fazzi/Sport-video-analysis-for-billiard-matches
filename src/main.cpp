#include <opencv2/opencv.hpp>
#include <iostream>
#include "../include/field_detection.h"
#include "../include/ball_detection.hpp"

using namespace cv;
using namespace std;
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path>" << std::endl;
        return -1;
    }

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return -1;
    }

	// Variables
		
	std::vector<cv::Mat> video_frames; // vector that stores each video frame
	cv::Mat frame;					   // temporary object to store each single frame
	cv::Mat cropped_field;			   // cropped image to perform ball detection
	vector<Point> field_points;		   // vector to store the points of the contour of the field
	std::vector<billiardBall> balls;   // vector to store object of balls


	// Read and store frames until video is complete
    while (cap.read(frame)) {
        video_frames.push_back(frame.clone());
	}
    
	// Release the video capture object
    cap.release();

	// -- FIELD DETECTION --
		
	// The first step involves to perform the field detection from the first frame of the selected video.
	// All the steps are located in the fiel field_detection.cpp
	// RETURN: the points of the corners of the billiard tables
	// PARAM: 
	// 1 - first video frame
	// 2 - output Mat object to store the image of the cropped table
	field_points = field_detection(video_frames[0], cropped_field);

	// -- BALL DETECTION --
	
	// Takes as input the cropped mask of the billiard table, to detect the balls
	// RETURN: vector of object billiardBall
	// PARAM: Mat object containing the cropped table
	balls =	ball_detection(cropped_field);

	

    cv::destroyAllWindows();
    return 0;
}

