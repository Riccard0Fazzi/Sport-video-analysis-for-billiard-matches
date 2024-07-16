#include "homography.h"

cv::Mat homography(const std::vector<cv::Point>& points)
{
	std::vector<cv::Point> tvp;
	if(points[3].x - points[2].x > points[1].x - points[0].x +20){
		// kp order HR, LR, LL, HL
		cv::Point two(453,79);
		tvp.push_back(two);
		cv::Point four(453,282);
		tvp.push_back(four);
		cv::Point three(62,282);
		tvp.push_back(three);
		cv::Point one(62,79);
		tvp.push_back(one);
		std::cout << "VERTICAL" << std::endl;
	}
	else{
		// kp order HL,HR,LL,LR
		cv::Point one(62,79);
		tvp.push_back(one);
		cv::Point two(453,79);
		tvp.push_back(two);
		cv::Point three(62,282);
		tvp.push_back(three);
		cv::Point four(453,282);
		tvp.push_back(four);
		std::cout << "HORIZONTAL" << std::endl;
	}

	// HOMOGRAPHY

	// Compute homography matrix
	cv::Mat H = cv::findHomography(points, tvp);

	// Output the computed homography matrix
	std::cout << "Homography Matrix:\n" << H << std::endl;

	return H;
}

void mapPoints(const cv::Mat H, const std::vector<cv::Point> points, cv::Scalar color)
{
	std::vector<cv::Point2f> points_to_transform;
	std::vector<cv::Point2f> transformed_points;
	for (const auto& point : points) {
		  points_to_transform.push_back(cv::Point2f(static_cast<float>(point.x), static_cast<float>(point.y)));
	}

	cv::perspectiveTransform(points_to_transform, transformed_points, H);
	// Read the image
	cv::Mat topView = imread("../data/Top_View.jpg", cv::IMREAD_ANYCOLOR);

	// Draw the points on image_a
	for (const auto& pt : transformed_points) {
		cv::circle(topView, pt, 4, color, -1);  // Red circles
	}
	cv::namedWindow("Projection");
	cv::imshow("Projection",topView);
	cv::waitKey(0);
}
