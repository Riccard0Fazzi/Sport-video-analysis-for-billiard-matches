#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include "../include/ball_detection.h"
#include "../include/field_detection.h"
#include "../include/trajectory_tracking.h"
#include "../include/homography.h"
using namespace cv;
using namespace std;

void tracking_balls(std::vector<cv::Mat>& all_video_frames, std::vector<billiardBall>& balls, cv::Mat& H) 
{
	vector<Ptr<TrackerCSRT>> trackers;
    vector<Rect> rois;
	vector<Mat> output_images;
	vector<vector<Point>> trajectory_lines;
	trajectory_lines.resize(balls.size());
	
	vector<billiardBall> balls_copy;
	vector<Mat> video_frames_copy;
	for(int j = 0; j < all_video_frames.size(); j++)
		video_frames_copy.push_back(all_video_frames[j].clone());

	for(int j = 0; j < balls.size(); j++)
	{
		balls_copy.push_back(balls[j]);
		rois.push_back(balls[j].box);
	}
    // Initialize trackers with the first frame and initial rectangles
    for (int i = 0; i < balls_copy.size(); i++) {
        Ptr<TrackerCSRT> tracker = TrackerCSRT::create();
		rois.push_back(balls_copy[i].box);
        tracker->init(video_frames_copy[0], balls_copy[i].box);
        trackers.push_back(tracker);
    }
	int big_index = 0;
    // Iterate through the video frames and update trackers ------------------------- for each frame
    for (const Mat& frame : video_frames_copy) {
        vector<Rect> tracked_rects;
        for (size_t i = 0; i < trackers.size(); ++i) {

            bool success = trackers[i]->update(frame, rois[i]);
            if (success) {
                tracked_rects.push_back(rois[i]); 
            } else {
                tracked_rects.push_back(Rect()); // Placeholder or empty Rect
            }
        }

        // Draw the rectangles on the current frame
        Mat display_frame = frame.clone();
		Mat overlayImage = imread("../data/Top_View.jpg",IMREAD_ANYCOLOR);
		vector<Point> old_points;
		vector<Point> new_points;
		for(int i = 0; i < tracked_rects.size(); i++)
			old_points.emplace_back(tracked_rects[i].x+tracked_rects[i].width/2,tracked_rects[i].y+tracked_rects[i].height/2);

		mapPoints(H,old_points,new_points);

		for(int i = 0; i < new_points.size(); i++)
			trajectory_lines[i].emplace_back(new_points[i]); // store the initial points of all coordinates

		// Draw the balls in the topview image

		Mat baseImage = display_frame;

		int index = 0;
		for(int i = 0; i < new_points.size(); i++)
		{
			circle(overlayImage,new_points[i],3,Scalar(0,255,0),5);
			for(const Point& point : trajectory_lines[index])
				circle(overlayImage,point,1,Scalar(255,255,255),1);
			index++;
		}
		
		Mat resizedOverlay;
		resize(overlayImage, resizedOverlay, cv::Size(baseImage.cols / 3, baseImage.rows / 3));

		//imshow("TopView",overlayImage);	
        for (const Rect& rect : tracked_rects) {
            if (rect.width > 0 && rect.height > 0) { // Ensure valid rectangles
                rectangle(display_frame, rect, Scalar(0, 0, 255),2);
            }

			// here we have to overlay the image

			// Resize overlay image if needed
			// For example, resize overlay image to be a quarter of the base image size
			// Define ROI on base image where overlay image will be placed (top-left corner here)
			Rect roi(0,display_frame.rows-resizedOverlay.rows, resizedOverlay.cols, resizedOverlay.rows); // Adjust position and size as needed

			// Check if ROI is within the base image bounds
			if (roi.x + roi.width > baseImage.cols || roi.y + roi.height > baseImage.rows) {
				std::cerr << "Overlay image exceeds base image bounds!" << std::endl;
			}

			// Place overlay image on base image
			Mat baseROI = baseImage(roi); // Define the region of interest on the base image
			resizedOverlay.copyTo(baseROI);  // Copy the overlay image onto the base image	
			}

			// Display the frame with tracked rectangles
			imshow("Multi-object Tracker", display_frame);
			if (waitKey(30) == 27) break;  // Exit on ESC key
			big_index++;
    }
	trajectory_lines.clear();


    destroyAllWindows();
}

