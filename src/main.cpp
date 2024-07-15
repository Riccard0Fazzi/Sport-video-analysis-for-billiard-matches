#include <opencv2/opencv.hpp>
#include <iostream>
#include "../include/field_detection.h"

using namespace cv;
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
	
	// vector that stores each video frame
	std::vector<cv::Mat> video_frames;
	// temporary object to store each single frame
	cv::Mat frame;

	// Read and store frames until video is complete
    while (cap.read(frame)) {
        video_frames.push_back(frame.clone()); // Use clone to copy the frame into vector
    }
    // Release the video capture object
    cap.release();
	
	// now i should apply the field detection from the first frame of the selected video
	// so i have to do something like field_detection(video_frames[0]);
	Mat tV = field_detection(video_frames[0]);

    cv::destroyAllWindows();
    return 0;
}

