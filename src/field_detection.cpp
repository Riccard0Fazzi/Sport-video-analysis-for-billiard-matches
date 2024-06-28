// In this workspace for the Billiard project
// Detection of the billiard table
// Created by crucio on 26/06/24.
// Task 1
// Command line argument:
/*
../data/first_frames_videos
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
        Mat img = src.clone();
        Mat proc;


        // PRE-PROCESSING
        // -------------------------------------------------------------------------------------------------------------

        // Turn to greyscale
        cvtColor(img, proc, COLOR_BGR2GRAY);

        // CONTRAST STRETCHING
        // Find minimum and maximum pixel values in the image
        double minVal, maxVal;
        minMaxLoc(proc, &minVal, &maxVal);

        // Apply contrast stretching
        proc = (proc - minVal) * (255.0 / (maxVal - minVal));

        // Apply histogram equalization
        equalizeHist(proc, proc);



        /*
        // BILATERAL FILTER
        Mat copy = proc.clone();
        int diameter = 9;   // Diameter of each pixel neighborhood
        double sigmaColor = 75;  // Filter sigma in the color space
        double sigmaSpace = 75;  // Filter sigma in the coordinate space
        bilateralFilter(copy, proc, diameter, sigmaColor, sigmaSpace);
        */
        // GAUSSIAN BLUR
        GaussianBlur(proc, proc, Size(5, 5), 0);

        // OTSU'S THRESHOLD
        double otsuThreshold = threshold(proc, proc, 0, 255, THRESH_BINARY | THRESH_OTSU);
        // TABLE DETECTION
        // -------------------------------------------------------------------------------------------------------------

        // CANNY EDGE DETECTOR
        Mat canny;
        int TL = otsuThreshold*5;
        int TH = otsuThreshold*6;
        Canny(proc, canny,TL, TH, 5, true);


        // HOUGH LINE TRANSFORM
        vector<Vec2f> lines; // vector to hold the results of the detection
        HoughLines(canny, lines, 1, CV_PI / 310, 160); // runs the actual detection
        canny.setTo(cv::Scalar(0, 0, 0));
        double Tolr = 50;
        double Tolt = 1;
        float rho = lines[0][0]+Tolr+1, theta = lines[0][1]+Tolt+1;

        // Draw the lines on image
        int count = 0;
        for (size_t i = 0; count < 4; i++) {
            std::cout << lines[i][1] << std::endl;
            // Extract rho and theta
            if(abs(rho - lines[i][0])>Tolr && abs(theta - lines[i][1])>Tolt ) {
                rho = lines[i][0], theta = lines[i][1];
                // Hybrid approach to draw only the best lines

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
                line(canny, pt1, pt2, Scalar(255, 255, 255), 3, LINE_AA);\
                count++;
            }
        }

        // HARRY'S CORNER
        // Step 3: Apply the Harris Corner Detector
        cv::Mat dst, dst_norm, dst_norm_scaled;
        dst = cv::Mat::zeros(canny.size(), CV_32FC1);

        int blockSize = 2; // Size of neighborhood considered for corner detection
        int apertureSize = 3; // Aperture parameter for the Sobel operator
        double k = 0.04; // Harris detector free parameter

        cv::cornerHarris(canny, dst, blockSize, apertureSize, k);

        // Step 4: Normalize the result
        cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
        cv::convertScaleAbs(dst_norm, dst_norm_scaled);

        // Step 5: Draw circles around detected corners
        for (int i = 0; i < dst_norm.rows; i++) {
            for (int j = 0; j < dst_norm.cols; j++) {
                if ((int)dst_norm.at<float>(i, j) > 200) {
                    cv::circle(img, cv::Point(j, i), 5, cv::Scalar(0, 0, 255), 2, 8, 0);
                }
            }
        }

        // SHOW OUTPUT
        // -------------------------------------------------------------------------------------------------------------

        // Show initial Canny image
        namedWindow("BILLIARD TABLE DETECTION");
        // Show initial Canny image
        imshow("BILLIARD TABLE DETECTION", canny);
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
