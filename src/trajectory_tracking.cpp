#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    // Load video or capture from camera
    VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video file." << std::endl;
        return 1;
    }

    Mat frame;
    cap >> frame;
    if (frame.empty()) {
        std::cerr << "Error: Could not read frame from video." << std::endl;
        return 1;
    }

    // Vector to store trackers and ROIs
    vector<Ptr<Tracker>> trackers;
    vector<Rect> rois;

    // Main tracking loop
    for (;;) {
        // Read frame
        cap >> frame;
        if (frame.empty()) break;

        // Select ROIs using mouse click
        if (trackers.empty()) {
            cout << "Select objects to track by clicking on them, then press any key to start tracking." << endl;
            while (true) {
                Mat clone = frame.clone();
                Rect roi = selectROI("Multi-object Tracker", clone);
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

        // Update trackers
        for (size_t i = 0; i < trackers.size(); ++i) {
            bool success = trackers[i]->update(frame, rois[i]);
            if (success) {
                rectangle(frame, rois[i], Scalar(0, 0, 255), 2);
            }/* else {
                cout << "Tracking failure detected for object " << i << endl;
            }*/
        }

        imshow("Multi-object Tracker", frame);

        int key = waitKey(30);
        if (key == 27) break;  // Exit on ESC key
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
