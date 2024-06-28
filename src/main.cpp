#include <opencv2/opencv.hpp>
#include <iostream>


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
	cv::namedWindow("Test_Frames");
	int i = 0;
	while(i < video_frames.size())
	{
		cv::imshow("Test_Frames",video_frames[i]);
		i++;
		cv::waitKey(0);
	}
    cv::destroyAllWindows();
    return 0;
}

