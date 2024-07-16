#include "homography.h"

void homography(std::vector<cv::Point> points)
{

        vector<Point> tvp;
        if(points[3].x - points[2].x > points[1].x - points[0].x +20){
            // kp order HR, LR, LL, HL
            Point two(453,79);
            tvp.push_back(two);
            Point four(453,282);
            tvp.push_back(four);
            Point three(62,282);
            tvp.push_back(three);
            Point one(62,79);
            tvp.push_back(one);
            std::cout << "VERTICAL" << std::endl;
        }
        else{
            // kp order HL,HR,LL,LR
            Point one(62,79);
            tvp.push_back(one);
            Point two(453,79);
            tvp.push_back(two);
            Point three(62,282);
            tvp.push_back(three);
            Point four(453,282);
            tvp.push_back(four);
            std::cout << "HORIZONTAL" << std::endl;
        }

        // HOMOGRAPHY

        // Compute homography matrix
        cv::Mat H = cv::findHomography(points, tvp);

        // Output the computed homography matrix
        std::cout << "Homography Matrix:\n" << H << std::endl;

        // Apply the homography to a point or an image
        // Define a point in the first image (image_a)


        std::vector<cv::Point2f> points_to_transform;
        // Define a point in the first image (image_a) that you want to plot in the second image (image_b)
        for (const auto& point : points) {
            points_to_transform.push_back(cv::Point2f(static_cast<float>(point.x), static_cast<float>(point.y)));
        }
        std::vector<cv::Point2f> transformed_points;

        // Transform the points to the second image (image_b)
        cv::perspectiveTransform(points_to_transform, transformed_points, H);
        // Read the image
        Mat topView = imread("../data/Top_View.jpg", IMREAD_ANYCOLOR);

		    // Draw the points on image_a
        for (const auto& pt : transformed_points) {
            cv::circle(topView, pt, 10, cv::Scalar(0, 0, 255), -1);  // Red circles
        }


        // SHOW OUTPUT
        // -------------------------------------------------------------------------------------------------------------

        // Show initial Canny image
        namedWindow("Homography");
        // Show initial Canny image
        imshow("Homography", topview);
        // Wait for trackbar adjustments and key press
        char key = waitKey(0);

        // Close all windows before moving to the next image
        destroyAllWindows();

}
