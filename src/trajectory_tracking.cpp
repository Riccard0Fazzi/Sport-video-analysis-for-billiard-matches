#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include "../include/ball_detection.hpp"
#include "../include/field_detection.h"
using namespace cv;
using namespace std;

void tracking_balls(vector<Mat>& all_video_frames, vector<billiardBall>& balls) 
{
	vector<Ptr<TrackerCSRT>> trackers;
    vector<Rect> rois;

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

    // Iterate through the video frames and update trackers
    for (const Mat& frame : video_frames_copy) {
        vector<Rect> tracked_rects;

        for (size_t i = 0; i < trackers.size(); ++i) {

            bool success = trackers[i]->update(frame, rois[i]);
            if (success) {
                tracked_rects.push_back(rois[i]);  // Convert Rect2d to Rect
            } else {
                tracked_rects.push_back(Rect()); // Placeholder or empty Rect
            }
        }

        // Draw the rectangles on the current frame
        Mat display_frame = frame.clone();
		Mat overlayImage = imread("../data/Top_View.jpg",IMREAD_ANYCOLOR);
        for (const Rect& rect : tracked_rects) {
		if (overlayImage.empty()) {
        std::cerr << "Could not open or find the images!" << std::endl;
    }	
			// Resize overlay image if needed
			// For example, resize overlay image to be a quarter of the base image size
			cv::Mat resizedOverlay;
			cv::resize(overlayImage, resizedOverlay, cv::Size(display_frame.cols / 2, display_frame.rows / 2));

			// Define ROI on base image where overlay image will be placed (top-left corner here)
			cv::Rect roi(0, display_frame.rows-resizedOverlay.rows,resizedOverlay.cols, resizedOverlay.rows); // Adjust position and size as needed

			// Check if ROI is within the base image bounds
			if (roi.x + roi.width > display_frame.cols || roi.y + roi.height > display_frame.rows) {
				std::cerr << "Overlay image exceeds base image bounds!" << std::endl;
			}

			// Place overlay image on base image
			cv::Mat baseROI = display_frame(roi); // Define the region of interest on the base image
			resizedOverlay.copyTo(baseROI);  // Copy the overlay image onto the base image

            if (rect.width > 0 && rect.height > 0) { // Ensure valid rectangles
                rectangle(display_frame, rect, Scalar(0, 0, 255), 2);
            }
        }

        // Display the frame with tracked rectangles
        imshow("Multi-object Tracker", display_frame);
        if (waitKey(30) == 27) break;  // Exit on ESC key
    }

    destroyAllWindows();
}

