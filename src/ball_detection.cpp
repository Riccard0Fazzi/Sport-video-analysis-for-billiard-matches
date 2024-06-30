// Created by Tommaso Tubaldo on 06/06/24 - Hours: 20
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv) {
    if (argc < 3) {
        std :: cout << "An image file and a image name with folder path should be provided!";
        return EXIT_FAILURE;
    }   // Checks for the correctness of the input values

    Mat img = imread(argv[1]);
    if (img.empty()) {
        std :: cout << "The image cannot be read!";
        return EXIT_FAILURE;
    }   // Check for the validity of the input image

    // Bilateral Filter after-conversion
    Mat filtered_gray_img;
    bilateralFilter(img,filtered_gray_img,7,90,300);

    std::string output_img_name13 = "/bilateralFilter.png";
    imwrite(argv[2]+output_img_name13,filtered_gray_img);

    // HSV conversion
    Mat hsv_img;
    cvtColor(filtered_gray_img, hsv_img, COLOR_BGR2HSV_FULL);

    // TABLE SEGMENTATION
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
    calcHist(&hsv_img,1,channels,Mat(),hist,3,histSize,ranges);

    // Find the bin with the maximum count
    double maxVal = 0;
    int maxIdx[3] = {0, 0, 0};
    minMaxIdx(hist, nullptr, &maxVal, nullptr, maxIdx);

    // Convert the bin index to HSV color
    int hBin = maxIdx[0], sBin = maxIdx[1], vBin = maxIdx[2];
    float hStep = 180.0f / hBins, sStep = 256.0f / sBins, vStep = 256.0f / vBins;
    Vec3b mostCommonColorHSV(hBin * hStep, sBin * sStep, vBin * vStep);

    // Create a mask for the most common color
    int h_thresh = 70;
    int s_thresh = 50;
    int v_thresh = 50;
    Scalar lowerBound(mostCommonColorHSV[0] - h_thresh, mostCommonColorHSV[1]-s_thresh, mostCommonColorHSV[2]-v_thresh);
    Scalar upperBound(mostCommonColorHSV[0] + h_thresh, mostCommonColorHSV[1]+s_thresh, mostCommonColorHSV[2]+v_thresh);
    Mat mask;
    inRange(hsv_img, lowerBound, upperBound, mask);

    // Invert the mask to remove the most common color
    cv::Mat invertedMask;
    cv::bitwise_not(mask, invertedMask);

    std::string output_img_name11 = "/mask.png";
    imwrite(argv[2]+output_img_name11,invertedMask);

    // Apply the mask to the original image
    cv::Mat result;
    img.copyTo(result, invertedMask);

    std::string output_img_name123 = "/mask_before_erosion.png";
    imwrite(argv[2]+output_img_name123,result);

    std::string output_img_name8 = "/mask.png";
    imwrite(argv[2]+output_img_name8,result);

    std::vector<Mat> res_channels;
    split(result,res_channels);
    threshold(res_channels[0],result,0,255,THRESH_BINARY);

    std::string output_img_name9 = "/mask_binary.png";
    imwrite(argv[2]+output_img_name9,result);

    // Apply erosion to the mask
    erode(result,result,getStructuringElement(MORPH_ELLIPSE,Size(3,3)),Point(-1,-1),5);
    std::string output_img_name4 = "/eroded_mask.png";
    imwrite(argv[2]+output_img_name4,result);

    dilate(result,result,getStructuringElement(MORPH_ELLIPSE,Size(3,3)),Point(-1,-1),5);
    // morphologyEx(result,result,MORPH_OPEN,getStructuringElement(MORPH_CROSS,Size(2,2)));

    std::string output_img_name3 = "/opened_mask.png";
    imwrite(argv[2]+output_img_name3,result);

    /*
    // Gamma Transform
    double gamma = 0.6;
    double c = pow(255,1-gamma);
    for (int i = 0; i < filtered_gray_img.rows; i++) {
        for (int j = 0; j < filtered_gray_img.cols; j++) {
            filtered_gray_img.at<uchar>(i,j) = c*pow(filtered_gray_img.at<uchar>(i,j),gamma);
        }
    }
    */

    Mat gray_img;
    // cvtColor(result,gray_img,COLOR_BGR2GRAY);
    result.copyTo(gray_img);

    std::string output_img_name21 = "/gray_img.png";
    imwrite(argv[2]+output_img_name21,gray_img);

    // Canny edge detector as test
    Mat canny_img;
    int thresh1_canny = 300;
    Canny(gray_img,canny_img, thresh1_canny/2,thresh1_canny);

    std::string output_img_name22 = "/cannyImg.png";
    imwrite(argv[2]+output_img_name22,canny_img);

    // Hough circle transformation is applied
    int thresh1 = 300;
    int thresh2 = 10;
    std::vector<Vec3f> circles;
    HoughCircles(gray_img,circles,HOUGH_GRADIENT,1,static_cast<float>(gray_img.rows)/36,thresh1,thresh2,gray_img.rows/42,gray_img.rows/8);

    if (!circles.empty()) {
        for (int i = 0; i < circles.size(); i++) {
            std::cout << circles[i,1] << ", " << circles[i,2] << ", " << circles[i,3] << '\n';
        }
    } else {
        std::cout << "No circles detected!!";
    }

    for(size_t i = 0; i < circles.size(); i++) {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        // draw the circle center
        circle(img, center, 1, Scalar(0,255,0), 1, LINE_AA);
        // draw the circle outline
        circle(img, center, radius, Scalar(0,0,255), 1, LINE_AA);
    }

    std::string output_img_name2 = "/circles.png";
    imwrite(argv[2]+output_img_name2,img);  // Saving the image on the desired folder

    return 0;
}