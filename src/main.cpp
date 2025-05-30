#include <opencv2/opencv.hpp>
#include <iostream>
#include "../include/field_detection.h"
#include "../include/ball_detection.h"
#include "../include/homography.h"
#include "../include/classification.h"
#include "../include/trajectory_tracking.h"
#include "../include/bounding_box.h"
using namespace cv;
using namespace std;
int main(int argc, char** argv) {

	if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path>" << std::endl;
        return -1;
    }

	string path = argv[1];
	string pattern = "*.mp4";
	vector<String> filenames;
	utils::fs::glob(path,pattern,filenames);


	std::vector<cv::Mat> video_frames; // vector that stores each video frame
	cv::Mat frame;					   // temporary object to store each single frame
	cv::Mat cropped_field;			   // cropped image to perform ball detection
	std::vector<cv::Point> field_points;		   // vector to store the points of the contour of the field
	std::vector<billiardBall> balls;   // vector to store object of balls
	std::vector<cv::Point> balls_coordinates;
	cv::Mat H;

	cv::VideoCapture cap;

	for(int i = 0;  i < filenames.size(); i++)
	{
		cap.open(filenames[i]);
		if (!cap.isOpened()) {
		std::cerr << "Error: Could not open video file." << std::endl;
		return -1;
		}

			
		// Read and store frames until video is complete
		while (cap.read(frame)) {
			video_frames.push_back(frame.clone());
		}
	
		
		// Release the video capture object
		cap.release();


		// -- Field detection --
			
		// The first step involves to perform the field detection from the first frame of the selected video.
		// All the steps are located in the fiel field_detection.cpp
		// RETURN: the points of the corners of the billiard tables
		// PARAM: 
		// 1 - first video frame
		// 2 - output Mat object to store the image of the cropped table
		field_points = field_detection(video_frames[0], cropped_field);

		// -- Ball detection --
		
		// Takes as input the cropped mask of the billiard table, to detect the balls
		// RETURN: vector of object billiardBall
		// PARAM: Mat object containing the cropped table
		balls =	ball_detection(cropped_field, field_points[field_points.size()-1]);

		classification(cropped_field,balls);

	Mat field_bounding_box = video_frames[0].clone();
		for(int i = 0; i < balls.size(); i++)
		{
			if(balls[i].id == 0) // white
				rectangle(field_bounding_box,balls[i].box,cv::Scalar(255,255,255),3);

			if(balls[i].id == 1) // black
				rectangle(field_bounding_box,balls[i].box,cv::Scalar(0,0,0),3);
			if(balls[i].id == 2) // solid
				rectangle(field_bounding_box,balls[i].box,cv::Scalar(255,0,0),3);
			if(balls[i].id == 3) // striped
			rectangle(field_bounding_box,balls[i].box,cv::Scalar(0,255,0),3);
		}

		imshow("FieldBB",field_bounding_box);
		waitKey(0);
		// create bounding box image

		Mat	H = computeHomography(field_points);
		tracking_balls(video_frames,balls,H);
        
		
		//-- Field homography --

		//mapPoints(H,field_points,cv::Scalar(255,0,0));
		// test homography for balls
		//for(int i = 0; i < balls.size(); i++)	balls_coordinates.emplace_back(balls[i].x,balls[i].y);	
		
		//mapPoints(H,balls_coordinates, cv::Scalar(0,255,0));


		video_frames.clear(); // vector that stores each video frame
		//balls_coordinates.clear();
		frame.release();					   // temporary object to store each single frame
		//H.release();
		cropped_field.release();			   // cropped image to perform ball detection
		field_points.clear();		   // vector to store the points of the contour of the field
		balls.clear();   // vector to store object of balls
		cv::destroyAllWindows();
		std::cout << "+------------------------------------[NEXT]--------------------------------------------+" << std::endl;
	}
		return 0;
}

