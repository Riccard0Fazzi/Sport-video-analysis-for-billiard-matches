// Created by Tommaso Tubaldo on 06/06/24 - Hours: 20
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

using namespace cv;

void imageCropping(const Mat& img, Mat& dest);
void contrastStretching(const Mat& img, Mat& dest, int brightness_increase);
void colorBasedSegmentation(const Mat& img, Mat& dest, std::vector<int> HSV_thresholds);

int main(int argc, char** argv) {
    if (argc < 3) {
        std :: cout << "An image file and a image name with folder path should be provided!";
        return EXIT_FAILURE;
    }   // Checks for the correctness of the input values

    std::string path = argv[1];
    std::string pattern = argv[3];
    std::vector<cv::String> filenames;
    utils::fs::glob(path,pattern,filenames);

    // Loop for every image
    for (size_t c = 0; c < filenames.size(); ++c) {
        // Read the image
        Mat img = imread(filenames[c], IMREAD_ANYCOLOR);
        // Safety check on the image returned
        if (img.empty()) // If filename is wrong, imread returns an empty Mat object
        {
            // Print an error message using cv::Error
            std::cerr << "Error: " << cv::format("Failed to load image! Error code: %d", cv::Error::StsError) << std::endl;
            exit(0);
        }

        imageCropping(img,img);

        // Bilateral Filter [d:7, sigmaColor:80, sigmaSpace:300]
        Mat filtered_img;
        bilateralFilter(img, filtered_img, 7, 80, 300);
        //std::string output_img_name13 = "/bilateralFilter.png";
        //imwrite(argv[2] + output_img_name13, filtered_img);

        // Contrast stretching used to enhance dark regions, and hence obtain a correct segmentation
        int brightness_increase = 50;
        contrastStretching(filtered_img, filtered_img, brightness_increase);
        //std::string output_img_name16 = "/equalized_HSV.png";
        //imwrite(argv[2] + output_img_name16, filtered_img);

        // Color-based segmentation applied to obtain the balls mask
        std::vector<int> HSV_thresholds = {20, 60, 60};   // (last used -> [70,50,50])
        Mat segmented_img;
        colorBasedSegmentation(filtered_img, segmented_img, HSV_thresholds);
        //std::string output_img_name8 = "/mask.png";
        //imwrite(argv[2] + output_img_name8, segmented_img);

        // Conversion to gray-scale and binary thresholding of the balls mask
        cvtColor(segmented_img, segmented_img, COLOR_BGR2GRAY);
        Mat binary_segmented_img;
        threshold(segmented_img, binary_segmented_img, 0, 255, THRESH_BINARY);
        //std::string output_img_name9 = "/mask_binary.png";
        //imwrite(argv[2] + output_img_name9, binary_segmented_img);

        // Morphological operators (CLOSING + OPENING) used to make more even the balls blobs
        morphologyEx(binary_segmented_img, binary_segmented_img, MORPH_CLOSE,
                     getStructuringElement(MORPH_ELLIPSE, Size(3, 3)), Point(-1, -1), 2);
        // Apply erosion to the mask
        erode(binary_segmented_img, binary_segmented_img, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)),
              Point(-1, -1), 3);
        //std::string output_img_name4 = "/eroded_mask.png";
        //imwrite(argv[2] + output_img_name4, binary_segmented_img);
        dilate(binary_segmented_img, binary_segmented_img, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)),
               Point(-1, -1), 3);
        // morphologyEx(result,result,MORPH_OPEN,getStructuringElement(MORPH_ELLIPSE,Size(3,3)),Point(-1,-1),3);
        //std::string output_img_name3 = "/opened_mask.png";
        //imwrite(argv[2] + output_img_name3, binary_segmented_img);

        // CIRCLE DETECTION from the binary mask
        // Canny edge detector as test
        Mat canny_img;
        int thresh1_canny = 300;
        Canny(binary_segmented_img, canny_img, static_cast<float>(thresh1_canny) / 2, thresh1_canny);
        //std::string output_img_name22 = "/cannyImg.png";
        //imwrite(argv[2] + output_img_name22, canny_img);

        // Hough circles transformation
        double min_distance_between_circles = static_cast<double>(binary_segmented_img.cols) / 40;
        int thresh1 = 300;
        int thresh2 = 11;
        double min_radius = static_cast<double>(std::max(binary_segmented_img.cols, binary_segmented_img.rows)) / 115;
        double max_radius = static_cast<double>(std::max(binary_segmented_img.cols, binary_segmented_img.rows)) / 35;
        std::vector<Vec3f> circles;
        HoughCircles(binary_segmented_img, circles, HOUGH_GRADIENT, 1, min_distance_between_circles, thresh1, thresh2,
                     min_radius, max_radius);

        // Print the locations of the founded balls
        if (!circles.empty()) {
            for (int i = 0; i < circles.size(); i++) {
                std::cout << circles[i, 1] << ", " << circles[i, 2] << ", " << circles[i, 3] << '\n';
            }
        } else {
            std::cout << "No circles detected!!";
        }

        // Visualize the detected balls in the original image
        for (size_t i = 0; i < circles.size(); i++) {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            // draw the circle center
            circle(img, center, 1, Scalar(0, 255, 0), 1, LINE_AA);
            // draw the circle outline
            circle(img, center, radius, Scalar(0, 0, 255), 1, LINE_AA);
        }

        std::string output_img_name2 = "/circles.png";
        imwrite(argv[2] + output_img_name2, img);  // Saving the image on the desired folder

    }
    return 0;
}

void contrastStretching(const Mat& img, Mat& dest, int brightness_increase) {
    // HSV conversion
    Mat hsv_img;
    cvtColor(img,hsv_img,COLOR_BGR2HSV_FULL);

    // Split the HSV image into channels
    std::vector<Mat> hsvChannels;
    split(hsv_img,hsvChannels);

    // Increase the brightness of the V channel
    for (int y = 0; y < hsvChannels[2].rows; y++) {
        for (int x = 0; x < hsvChannels[2].cols; x++) {
            if (hsvChannels[2].at<uchar>(y,x) != 0) {
                uchar& pixel = hsvChannels[2].at<uchar>(y, x);
                pixel = saturate_cast<uchar>(pixel + brightness_increase * (255 - pixel) / 255);
            }
        }
    }

    // Merge the channels back into one image
    Mat equalizedHsvImage;
    merge(hsvChannels,hsv_img);

    cvtColor(hsv_img,dest,COLOR_HSV2BGR_FULL);
}

void colorBasedSegmentation(const Mat& img, Mat& dest, std::vector<int> HSV_thresholds) {
    // HSV conversion
    Mat hsv_img;
    cvtColor(img,hsv_img,COLOR_BGR2HSV_FULL);

    // Create a histogram with 30 bins for Hue, 32 bins for Saturation, and 32 bins for Value
    int h_bins = 30, s_bins = 32, v_bins = 32;
    int hist_size[] = {h_bins, s_bins, v_bins};

    // Hue varies from 0 to 179, Saturation and Value from 0 to 255
    float h_range[] = {0, 180};
    float s_range[] = {0, 256};
    float v_range[] = {0, 256};
    const float* ranges[] = {h_range, s_range, v_range};

    // Use the 0-th, 1-st, and 2-nd channels
    int channels[] = {0, 1, 2};

    Mat hist;
    calcHist(&hsv_img,1,channels,Mat(),hist,3,hist_size,ranges);

    // Find the bin with the maximum count
    double max_val = 0;
    int max_idx[3] = {0, 0, 0};
    minMaxIdx(hist, nullptr, &max_val, nullptr, max_idx);

    // Convert the bin index to HSV color
    int h_bin = max_idx[0], s_bin = max_idx[1], v_bin = max_idx[2];
    float h_step = 180.0f / h_bins, s_step = 256.0f / s_bins, v_step = 256.0f / v_bins;
    Vec3b most_common_color(h_bin * h_step, s_bin * s_step, v_bin * v_step);

    // Create a mask for the most common color
    Scalar lower_bound(most_common_color[0] - HSV_thresholds[0], most_common_color[1] - HSV_thresholds[1], most_common_color[2] - HSV_thresholds[2]);
    Scalar upper_bound(most_common_color[0] + HSV_thresholds[0], most_common_color[1] + HSV_thresholds[1], most_common_color[2] + HSV_thresholds[2]);
    Mat mask;
    inRange(hsv_img,lower_bound,upper_bound,mask);

    // Invert the mask to remove the most common color
    Mat inverted_mask;
    bitwise_not(mask,inverted_mask);

    // Apply the mask to the original image
    img.copyTo(dest,inverted_mask);
}