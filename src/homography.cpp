#include "ball_detection.hpp"
#include "homography.h"

cv::Mat computeHomography(const std::vector<cv::Point>& points) {
	std::vector<cv::Point> tvp;
	std::vector<cv::Point> points_copy;
	for(int i = 0; i < points.size()-1; i++)
		points_copy.push_back(points[i]);
    std::vector<cv::Point> top_view_corners;
    // Define the 2D top-view corners in order: Upper-Left, Upper-Right, Bottom-Left, Bottom-Right
    top_view_corners = {cv::Point (62,79),
                        cv::Point (453,79),
                        cv::Point (62,282),
                        cv::Point (453,282)};

    // If the table satisfies the following condition, then it is selected as vertical table
	if(points[3].x - points[2].x > points[1].x - points[0].x +20){
		// corners order: Upper-Right, Bottom-Right, Upper-Left, Bottom-Left
		tvp.push_back(top_view_corners[1]);
		tvp.push_back(top_view_corners[3]);
		tvp.push_back(top_view_corners[0]);
		tvp.push_back(top_view_corners[2]);
	} else {
		// corners order Upper-Left, Upper-Right, Bottom-Left, Bottom-Right
		tvp.push_back(top_view_corners[0]);
		tvp.push_back(top_view_corners[1]);
		tvp.push_back(top_view_corners[2]);
		tvp.push_back(top_view_corners[3]);
	}

	// Compute homography matrix
	
	return cv::findHomography(points_copy, tvp);
}

void mapPoints(const cv::Mat H, const std::vector<cv::Point>& input_points, std::vector<cv::Point>& output_points) {
    std::vector<cv::Point2f> points_to_transform;
    std::vector<cv::Point2f> transformed_points;

    // Get the points to transform
    for (const auto& point : input_points) {
        points_to_transform.emplace_back(static_cast<float>(point.x), static_cast<float>(point.y));
    }

    // Apply the perspective transformation H
    cv::perspectiveTransform(points_to_transform, transformed_points, H);

    // Place the transformed points in the output vector
    for (const auto& point: transformed_points) {
        output_points.emplace_back(static_cast<int>(point.x), static_cast<int>(point.y));
    }
}

void applyPerspectiveTransform(const std::vector<cv::Point>& field_points, const std::vector<cv::Point>& balls_coords, std::vector<cv::Point>& transformed_balls_coords) {
    // Compute the homography transform
    cv::Mat H = computeHomography(field_points);

    // Define the x and y offset, due to the different origin between the field and the balls coordinates
    int offset_x;
    int offset_y;
    // Offset in the x direction is given by the x coordinates of the bottom left corner
    offset_x = field_points[2].x;
    // Offset in the y direction is given by the y coordinates of the upper left corner
    offset_y = field_points[0].y;

    // Apply the translation
    std::vector<cv::Point> translated_balls_coords;
    for (const auto& ball: balls_coords) {
        translated_balls_coords.emplace_back(ball.x + offset_x, ball.y + offset_y);
    }

    // Apply the transformation on the ball coordinates
    mapPoints(H,translated_balls_coords,transformed_balls_coords);
}
