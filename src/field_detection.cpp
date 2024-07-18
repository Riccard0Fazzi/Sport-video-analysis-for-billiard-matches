#include "../include/field_detection.h"
#include "../include/ball_detection.hpp"

using namespace cv;
using namespace std;

std::vector<cv::Point> field_detection(const cv::Mat& inputImage, Mat & cropped_field)
{
	//--  VARIABLES --
 	Mat src; 
	Mat img;
	Mat hsv_img;
	Mat filtered_gray_img;
	Mat hist;
	Mat mask;
	Mat invertedMask;
	Mat canny;
	int TL = 150;
	int TH = 200;

	std::vector<cv::Point> center_points;
	// histogram variables
	int hBins = 30, sBins = 32, vBins = 32;
	// Create a histogram with 30 bins for Hue, 32 bins for Saturation, and 32 bins for Value
	int histSize[] = {hBins, sBins, vBins};
	// Hue varies from 0 to 179, Saturation and Value from 0 to 255
	float h_range[] = {0, 180};
	float s_range[] = {0, 256};
	float v_range[] = {0, 256};
	const float* ranges[] = {h_range, s_range, v_range};
	// Use the 0-th, 1-st, and 2-nd channels
	int channels[] = {0, 1, 2};
	double maxVal = 0;
	int maxIdx[3] = {0, 0, 0};
	int h_thresh = 30;//20
	int s_thresh = 70;//80
	int v_thresh = 90;//80
	int hBin, sBin, vBin;
	float hStep, sStep, vStep;
	
	// clustering variables
	int K = 4;  // Number of clusters
	Mat labels, centers;
	vector<Vec2f> lines; // vector to hold the results of the detection
	
	// variables to create the cropped field
	cv::Mat temp = cv::Mat::zeros(inputImage.size(), CV_8UC1);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;



    // Safety check on the command line argument
    if(inputImage.empty()) {
        std::cout << "WARNING: An image shall be provided." << std::endl;
        exit(0);
    }

	// Read the image
	src = inputImage.clone();
	//namedWindow("CropField");
	//imshow("CropField",src);	
	//waitKey(0);



	// PRE-PROCESSING       

	// HSV conversion
	img = src.clone();
	cvtColor(src, hsv_img, COLOR_BGR2HSV_FULL);

	// Bilateral Filter after-conversion
	bilateralFilter(hsv_img,filtered_gray_img,5,50,20);

	calcHist(&filtered_gray_img,1,channels,Mat(),hist,3,histSize,ranges);

	// Find the bin with the maximum count
	minMaxIdx(hist, nullptr, &maxVal, nullptr, maxIdx);

	// Convert the bin index to HSV color
	hBin = maxIdx[0], sBin = maxIdx[1], vBin = maxIdx[2];
	hStep = 180.0f / hBins, sStep = 256.0f / sBins, vStep = 256.0f / vBins;
	Vec3b mostCommonColorHSV(hBin * hStep, sBin * sStep, vBin * vStep);

	// Create a mask for the most common color
	Scalar lowerBound(mostCommonColorHSV[0] - h_thresh, mostCommonColorHSV[1]-s_thresh, mostCommonColorHSV[2]-v_thresh);
	Scalar upperBound(mostCommonColorHSV[0] + h_thresh, mostCommonColorHSV[1]+s_thresh, mostCommonColorHSV[2]+v_thresh);
	inRange(filtered_gray_img, lowerBound, upperBound, mask);

	// Invert the mask to remove the most common color
	cv::bitwise_not(mask, invertedMask);
	

	// TABLE DETECTION -------------

	// Canny edge detector
	Canny(mask, canny,TL, TH, 3);

	// Hough line transform
	HoughLines(canny, lines, 1, CV_PI / 180, 110); // runs the actual detection / 110

	canny.setTo(cv::Scalar(0, 0, 0));
	Mat backup = canny.clone();

	// Find best four lines
	// Convert lines to a format suitable for k-means clustering
	Mat data(lines.size(), 2, CV_32F);
	for (size_t i = 0; i < lines.size(); ++i) {
		data.at<float>(i, 0) = lines[i][0];  // rho
		data.at<float>(i, 1) = lines[i][1];  // theta
	}

	TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0);
	kmeans(data, K, labels, criteria, 3, KMEANS_RANDOM_CENTERS, centers);
	// Create structures to detect corners
	vector<Mat> corners(centers.rows, canny.clone());
	// Initialize the corners vector with images of the same size as 'canny'

	// Clone 'canny' image for each element in corners vector
	for (int i = 0; i < centers.rows; ++i) {
		corners[i] = canny.clone();
	}
	// Draw the lines on image
	for (size_t i = 0; i < centers.rows; i++) {

		// Extract rho and theta
		double rho = centers.at<float>(i, 0), theta = centers.at<float>(i, 1);

		// Calculate direction vector
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		// Calculate two points far apart on the line
		Point pt1, pt2;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));

		// Draw the line
		line(canny, pt1, pt2, Scalar(255, 255, 255), 1, LINE_AA);
		// Draw the line on corners image
		line(corners[i], pt1, pt2, Scalar(255, 255, 255), 1, LINE_AA);
	}
	// Create the FIELD MASK

	for (size_t u = 0; u < img.rows; u++ ){
		for (size_t v = 0; v < img.cols; v++ ){
			if(canny.at<uchar>(u,v)<255){
				img.at<Vec3b>(u,v)[0] = 0;
				img.at<Vec3b>(u,v)[1] = 0;
				img.at<Vec3b>(u,v)[2] = 0;
			}
		}
	}
	//imshow("CropField",img);	
	//waitKey(0);

	// FIND INTERSECTIONS
	vector<Point> kp;
	for (int i = 0; i < corners[0].rows; i++) {
		for (int j = 0; j < corners[0].cols; j++) {
			int count = 0;
			for (int k = 0; k < 4; k++) 
				if (corners[k].at<uchar>(i,j) > 0) count++;

			if (count >= 2) {  // If at least two images have the value 255 at (i, j)
				Point p(j, i);  // Use (x, y) format
				//circle(img, p, 2, Scalar(0,255,255), -1);  // Draw red circles at the intersection points
				//circle(img, p, 10, Scalar(0,0,255), 1);  // Draw red circles at the intersection
				kp.push_back(p);
				//std::cout << "Point(x=" << p.x << ", y=" << p.y << ")" << std::endl;
			}
		}
	}
	// FIND BEST FOUR POINTS
	// Convert keypoints to a format suitable for k-means clustering
	cv::Mat points(kp.size(), 2, CV_32F);
	for (size_t i = 0; i < kp.size(); ++i) {
		points.at<float>(i, 0) = kp[i].x;
		points.at<float>(i, 1) = kp[i].y;
	}

	//cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0);
	cv::kmeans(points, K, labels, criteria, 3, cv::KMEANS_RANDOM_CENTERS, centers);

	// Convert centers to vector of points
	for (int i = 0; i < centers.rows; ++i) {
		cv::Point p(static_cast<int>(centers.at<float>(i, 0)), static_cast<int>(centers.at<float>(i, 1)));
		center_points.push_back(p);
	}

	// Sort the points to ensure they are in the order: upper left, upper right, lower left, lower right
	std::sort(center_points.begin(), center_points.end(), [](const cv::Point& a, const cv::Point& b) {
		return a.y < b.y || (a.y == b.y && a.x < b.x);
	});

	// Make sure upper points are before lower points
	if (center_points[0].x > center_points[1].x) std::swap(center_points[0], center_points[1]);
	if (center_points[2].x > center_points[3].x) std::swap(center_points[2], center_points[3]);

	// Print the sorted points
	/*std::cout << "Sorted Points (Upper Left, Upper Right, Lower Left, Lower Right):" << std::endl;
	for (const auto& point : center_points) {
		std::cout << "Point(x=" << point.x << ", y=" << point.y << ")" << std::endl;
	}
*/
	// Draw circles for the first point in center_points
	for (const auto& point : center_points) {
		cv::circle(temp, point, 2, cv::Scalar(255), -1);  // Draw yellow circles at the intersection points
		cv::circle(temp, point, 10, cv::Scalar(0, 0, 255), 1);   // Draw red circles at the intersection
	}

	int thickness = 1;
	cv::line(temp, center_points[0], center_points[1], cv::Scalar(255), thickness); // Draw line 0-1
	cv::line(temp, center_points[1], center_points[3], cv::Scalar(255), thickness); // Draw line 1-2
	cv::line(temp, center_points[3], center_points[2], cv::Scalar(255), thickness); // Draw line 2-3
	cv::line(temp, center_points[2], center_points[0], cv::Scalar(255), thickness); // Draw line 3-0

	findContours(temp, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	//imshow("CropField",temp);	
	//waitKey(0);

	// Draw the mask
	drawContours(temp, contours, 1, Scalar(255), FILLED);

	img = inputImage.clone();

	// Create the FIELD MASK

	for (size_t u = 0; u < img.rows; u++ ){
		for (size_t v = 0; v < img.cols; v++ ){
			if(temp.at<uchar>(u,v)<255){
				img.at<Vec3b>(u,v)[0] = 0;
				img.at<Vec3b>(u,v)[1] = 0;
				img.at<Vec3b>(u,v)[2] = 0;
			}
		}
	}

	//CROP FIELD MASK
	Rect bounding_box;
	bounding_box = boundingRect(contours[1]);
	// Crop the image using the bounding box
	cropped_field = img(bounding_box);
	Mat temp_cropped_field;
	cvtColor(cropped_field,temp_cropped_field,COLOR_BGR2HSV_FULL);
	Vec3b most_common_color;
	Rect square_rect((temp_cropped_field.cols-60)/2,(temp_cropped_field.rows-60)/2,60,60);
	mostCommonColor(temp_cropped_field(square_rect),most_common_color);	
	Mat rgb_mat(100, 100, CV_8UC3, most_common_color);	
	cvtColor(rgb_mat,rgb_mat,COLOR_HSV2BGR_FULL);

	/*namedWindow("Rect");
	imshow("Rect",temp_cropped_field(square_rect));
	waitKey(0);
*/
	Mat mask_black;
		inRange(cropped_field,Scalar(0,0,0),Scalar(0,0,0),mask_black);
	
	cropped_field.setTo(rgb_mat.at<Vec3b>(1,1),mask_black);



	//imshow("CropField",cropped_field);
	//waitKey(0);

	return center_points;
}

