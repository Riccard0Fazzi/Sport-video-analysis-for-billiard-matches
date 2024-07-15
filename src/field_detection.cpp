// In this workspace for the Billiard project
// Detection of the billiard table using HISTOGRAM THRESHOLDING
// Created by crucio on 30/06/24.
// Command line argument:
/*
/Users/crucio/CLionProjects/ProjectWorkSpace/Project_images
*.png
*/

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

using namespace cv;
using namespace std;



int main (int argc, char** argv)
{
    // LOAD IMAGES
    // -----------------------------------------------------------------------------------------------------------------


    // Safety check on the command line argument
    if(argc < 2) {
        std::cout << "WARNING: An image filename shall be provided." << std::endl;
        return EXIT_FAILURE;
    }
    string path = argv[1];
    string pattern = argv[2];
    vector<cv::String> filenames;
    utils::fs::glob(path, pattern, filenames);

    // Load Top-View image
    // Read the image
    Mat topView = imread("/Users/crucio/CLionProjects/ProjectWorkSpace/Project_images/Top_View.jpg", IMREAD_ANYCOLOR);
    // Safety check on the image returned
    if (topView.empty()) // If filename is wrong, imread returns an empty Mat object
    {
        // Print an error message using cv::Error
        std::cerr << "Error: " << cv::format("Failed to load image! Error code: %d", cv::Error::StsError) << std::endl;
        exit(0);
    }
    // print

    // LOOP FOR EVERY IMAGE IN THE DATASET
    // -----------------------------------------------------------------------------------------------------------------


    for (size_t c = 0; c < filenames.size(); ++c) {

        // READ IMAGE
        // -------------------------------------------------------------------------------------------------------------


        // Read the image
        Mat src = imread(filenames[c], IMREAD_ANYCOLOR);
        // Safety check on the image returned
        if (src.empty()) // If filename is wrong, imread returns an empty Mat object
        {
            // Print an error message using cv::Error
            std::cerr << "Error: " << cv::format("Failed to load image! Error code: %d", cv::Error::StsError) << std::endl;
            exit(0);
        }



        // PRE-PROCESSING
        // -------------------------------------------------------------------------------------------------------------


        // HSV conversion
        Mat img = src.clone();
        Mat hsv_img;
        cvtColor(src, hsv_img, COLOR_BGR2HSV_FULL);

        // Bilateral Filter after-conversion
        Mat filtered_gray_img;
        bilateralFilter(hsv_img,filtered_gray_img,5,50,20);

        // Create a histogram with 30 bins for Hue, 32 bins for Saturation, and 32 bins for Value
        int hBins = 30, sBins = 32, vBins = 32;
        int histSize[] = {hBins, sBins, vBins};

        // Hue varies from 0 to 179, Saturation and Value from 0 to 255
        float h_range[] = {0, 180};
        float s_range[] = {0, 256};
        float v_range[] = {0, 256};
        const float* ranges[] = {h_range, s_range, v_range};

        // Use the 0-th, 1-st, and 2-nd channels
        int channels[] = {0, 1, 2};

        Mat hist;
        calcHist(&filtered_gray_img,1,channels,Mat(),hist,3,histSize,ranges);

        // Find the bin with the maximum count
        double maxVal = 0;
        int maxIdx[3] = {0, 0, 0};
        minMaxIdx(hist, nullptr, &maxVal, nullptr, maxIdx);

        // Convert the bin index to HSV color
        int hBin = maxIdx[0], sBin = maxIdx[1], vBin = maxIdx[2];
        float hStep = 180.0f / hBins, sStep = 256.0f / sBins, vStep = 256.0f / vBins;
        Vec3b mostCommonColorHSV(hBin * hStep, sBin * sStep, vBin * vStep);

        // Create a mask for the most common color
        int h_thresh = 20;//80
        int s_thresh = 80;//80
        int v_thresh = 80;//80
        Scalar lowerBound(mostCommonColorHSV[0] - h_thresh, mostCommonColorHSV[1]-s_thresh, mostCommonColorHSV[2]-v_thresh);
        Scalar upperBound(mostCommonColorHSV[0] + h_thresh, mostCommonColorHSV[1]+s_thresh, mostCommonColorHSV[2]+v_thresh);
        Mat mask;
        inRange(filtered_gray_img, lowerBound, upperBound, mask);

        // Invert the mask to remove the most common color
        cv::Mat invertedMask;
        cv::bitwise_not(mask, invertedMask);



        // TABLE DETECTION
        // -------------------------------------------------------------------------------------------------------------




        // CANNY EDGE DETECTOR
        Mat canny;
        int TL = 150;
        int TH = 200;
        Canny(mask, canny,TL, TH, 3);


        // HOUGH LINE TRANSFORM
        vector<Vec2f> lines; // vector to hold the results of the detection
        HoughLines(canny, lines, 1, CV_PI / 180, 110); // runs the actual detection / 110

        canny.setTo(cv::Scalar(0, 0, 0));
        Mat backup = canny.clone();




        // FIND BEST FOUR LINES
        // Convert lines to a format suitable for k-means clustering
        Mat data(lines.size(), 2, CV_32F);
        for (size_t i = 0; i < lines.size(); ++i) {
            data.at<float>(i, 0) = lines[i][0];  // rho
            data.at<float>(i, 1) = lines[i][1];  // theta
        }
        int K = 4;  // Number of clusters
        Mat labels, centers;
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

        /*
        // Find contours and hierarchy
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(canny, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        // Draw the mask
        canny.setTo(cv::Scalar(0, 0, 0));
        drawContours(canny, contours, 1, Scalar(255), FILLED);

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

        // CROP FIELD MASK
        Rect bounding_box;
        Mat dest;
        bounding_box = boundingRect(contours[1]);
        // Crop the image using the bounding box
        dest = img(bounding_box);

        // save FIELD MASK
        // Specify the output path
        std::string outputPath = "/Users/crucio/CLionProjects/ProjectWorkSpace/Field_Masks/"+to_string(c+1)+".png";

        // Save the image to the specified path
        bool isSuccess = cv::imwrite(outputPath, dest);
        if (!isSuccess)
        {
            std::cerr << "Error: Could not save the image to the specified path." << std::endl;
            return -1;
        }

        std::cout << "Image successfully saved to " << outputPath << std::endl;
         */


        // FIND INTERSECTIONS

        vector<Point> kp;
        for (int i = 0; i < corners[0].rows; i++) {
            for (int j = 0; j < corners[0].cols; j++) {
                int count = 0;

                for (int k = 0; k < 4; k++) {
                    if (corners[k].at<uchar>(i,j) > 0) {
                        count++;
                    }
                }
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
        std::vector<cv::Point> center_points;
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
        std::cout << "Sorted Points (Upper Left, Upper Right, Lower Left, Lower Right):" << std::endl;
        for (const auto& point : center_points) {
            std::cout << "Point(x=" << point.x << ", y=" << point.y << ")" << std::endl;
        }

        // SHOW OUTPUT
        // Assuming 'img' is defined and used for display
        // Draw circles for the first point in center_points
        for (const auto& point : center_points) {
            cv::circle(img, point, 2, cv::Scalar(0, 255, 255), -1);  // Draw yellow circles at the intersection points
            cv::circle(img, point, 10, cv::Scalar(0, 0, 255), 1);   // Draw red circles at the intersection
        }
        /*
        // Print the cluster centers
        std::cout << "Cluster centers:" << std::endl;
        for (int i = 0; i < centers.rows; ++i) {
            Point p(static_cast<int>(centers.at<float>(i, 0)), static_cast<int>(centers.at<float>(i, 1)));
            circle(img, p, 2, Scalar(0,255,255), -1);  // Draw red circles at the intersection points
            circle(img, p, 10, Scalar(0,0,255), 1);  // Draw red circles at the intersection

            std::cout << "Point(x=" << p.x << ", y=" << p.y << ")" << std::endl;
        }*/

        vector<Point> tvp;
        if(center_points[3].x - center_points[2].x > center_points[1].x - center_points[0].x +20){
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
        cv::Mat H = cv::findHomography(center_points, tvp);

        // Output the computed homography matrix
        std::cout << "Homography Matrix:\n" << H << std::endl;

        // Apply the homography to a point or an image
        // Define a point in the first image (image_a)


        std::vector<cv::Point2f> points_to_transform;
        // Define a point in the first image (image_a) that you want to plot in the second image (image_b)
        for (const auto& point : center_points) {
            points_to_transform.push_back(cv::Point2f(static_cast<float>(point.x), static_cast<float>(point.y)));
        }
        std::vector<cv::Point2f> transformed_points;

        // Transform the points to the second image (image_b)
        cv::perspectiveTransform(points_to_transform, transformed_points, H);
        // Read the image
        Mat topView = imread("/Users/crucio/CLionProjects/ProjectWorkSpace/Project_images/Top_View.jpg", IMREAD_ANYCOLOR);

        // Draw the points on image_a
        for (const auto& pt : transformed_points) {
            cv::circle(topView, pt, 10, cv::Scalar(0, 0, 255), -1);  // Red circles
        }






        // SHOW OUTPUT
        // -------------------------------------------------------------------------------------------------------------

        // Show initial Canny image
        namedWindow("BILLIARD TABLE DETECTION");
        // Show initial Canny image
        imshow("BILLIARD TABLE DETECTION", topView);
        // Wait for trackbar adjustments and key press
        char key = waitKey(0);

        // Check if 'q' is pressed to quit, otherwise continue
        if (key == 'q' || key == 'Q')
            break;

        // Close all windows before moving to the next image
        destroyAllWindows();
    }

    return EXIT_SUCCESS;
}

