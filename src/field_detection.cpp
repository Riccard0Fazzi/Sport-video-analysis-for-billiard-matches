// In this workspace for the Billiard project
// Detection of the billiard table using HISTOGRAM THRESHOLDING
// Created by crucio on 30/06/24.
// Command line argument:
/*
../data/Field_Masks
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




        vector<Vec2f> lines; // vector to hold the results of the detection
        HoughLines(canny, lines, 1, CV_PI / 180, 110); // runs the actual detection / 110

        canny.setTo(cv::Scalar(0, 0, 0));

        // Draw the rlines on image
        for (size_t i = 0; i < lines.size(); i++) {

            // Extract rho and theta
            double rho = lines[i][0], theta = lines[i][1];

            // Calculate direction vector
            double a = cos(theta), b = sin(theta);
            double x0 = a * rho, y0 = b * rho;
            // Calculate two points far apart on the line
            Point pt1, pt2;
            pt1.x = cvRound(x0 + 1000 * (-b));
            pt1.y = cvRound(y0 + 1000 * (a));
            pt2.x = cvRound(x0 - 1000 * (-b));
            pt2.y = cvRound(y0 - 1000 * (a));

        }


        // Find the 4 best lines
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
        }



        // Draw the lines on image


        // Find contours and hierarchy
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(canny, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        // Draw the mask
        canny.setTo(cv::Scalar(0, 0, 0));
        // Loop over all contours
        for (size_t i = 0; i < contours.size(); i++) {
            drawContours(canny, contours, 1, Scalar(255), FILLED);
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

        // save FIELD MASK
        /*
        // Specify the output path
        std::string outputPath = "/Users/crucio/CLionProjects/ProjectWorkSpace/Field_Masks/"+to_string(c+1)+".png";

        // Save the image to the specified path
        bool isSuccess = cv::imwrite(outputPath, img);
        if (!isSuccess)
        {
            std::cerr << "Error: Could not save the image to the specified path." << std::endl;
            return -1;
        }

        std::cout << "Image successfully saved to " << outputPath << std::endl;
         */









        /*
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
                    cv::circle(canny, cv::Point(j, i), 5, cv::Scalar(0, 0, 255), 2, 8, 0);
                }
            }
        }
        */




        // SHOW OUTPUT
        // -------------------------------------------------------------------------------------------------------------

        // Show initial Canny image
        namedWindow("BILLIARD TABLE DETECTION");
        // Show initial Canny image
        imshow("BILLIARD TABLE DETECTION", img);
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
