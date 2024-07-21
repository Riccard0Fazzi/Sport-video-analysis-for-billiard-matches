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
    // Vector to store trackers and ROIs
    vector<Ptr<Tracker>> trackers;
    vector<Rect> rois;
	Mat frame;
	cout << "Number of balls: " << balls.size() << endl;
    // Main tracking loop
    for (int i = 0; i < all_video_frames.size(); i++) {
        // Read frame
		frame = all_video_frames[i].clone();
        if (frame.empty()) break;

        // Select ROIs using mouse click
        for(int j = 0; j < balls.size(); j++)
		{		
			Mat clone = frame.clone();

//cout << "Roi data: X: " << balls[i].box.x << " Y: " << balls[i].box.y << " Width: "<< balls[i].box.width << " Height: " << balls[i].box.height << endl;
                Rect roi = balls[i].box;
                if (roi.width > 0 && roi.height > 0) {
                    Ptr<Tracker> tracker = TrackerCSRT::create();
                    tracker->init(frame, roi);
                    trackers.push_back(tracker);
                    rois.push_back(roi);
                } else {
                    break;
                }
            }
		}
        

		cout << "Number of rois: " << rois.size() << endl;
		cout << "Number of trackers: " << rois.size() << endl;
        // Update trackers
        for (size_t i = 0; i < trackers.size(); ++i) {
            bool success = trackers[i]->update(all_video_frames[i], rois[i]);
            if (success) {
                rectangle(all_video_frames[i], rois[i], Scalar(0, 0, 255), 2);
            } else {
                cout << "Tracking failure detected for object " << i << endl;
            }
        }

        imshow("Multi-object Tracker", frame);

        int key = waitKey(30);
        if (key == 27) break;  // Exit on ESC key
}
    destroyAllWindows();

}
