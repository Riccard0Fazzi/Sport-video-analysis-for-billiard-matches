// Created by Tommaso Tubaldo on 06/06/24 - Hours: 70
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include "ball_detection.hpp"

using namespace cv;

int main(int argc, char** argv) {
    // Check if the input values are correct
    if (argc < 3) {
        std :: cout << "An image file and a image name with folder path should be provided!";
        return EXIT_FAILURE;
    }

    std::string path = argv[1];
    std::string pattern = argv[3];
    std::vector<cv::String> filenames;
    utils::fs::glob(path,pattern,filenames);

    // Loop for every image
    for (size_t c = 0; c < filenames.size(); ++c) {
        std::string num = std::to_string(c);    // Numerical value for filename indexing

        // Read the image
        Mat img = imread(filenames[c], IMREAD_ANYCOLOR);
        // Safety check on the image returned
        if (img.empty()) // If filename is wrong, imread returns an empty Mat object
        {
            // Print an error message using cv::Error
            std::cerr << "Error: " << format("Failed to load image! Error code: %d", cv::Error::StsError) << std::endl;
            exit(0);
        }

        // Detection of the billiard balls
        std::vector<Vec3f> circles;
        ballDetection(img,circles);

        // Draw the detected circles
        Mat circles_img;
        drawCircles(img,circles_img,circles);

        // Print the detected circles
        std::string output_img_name0 = "/circles";
        imwrite(argv[2] + output_img_name0 + num + ".png",circles_img);
    }
    return 0;
}

void mostCommonColor(const Mat& img, Vec3b& most_common_color) {
    // Calculate the histogram of the image
    int h_bins = 30, s_bins = 32, v_bins = 32;
    int hist_size[] = {h_bins, s_bins, v_bins};
    float h_range[] = {0, 180};
    float s_range[] = {0, 256};
    float v_range[] = {0, 256};
    const float* ranges[] = {h_range, s_range, v_range};
    int channels[] = {0, 1, 2};

    // Create a mask to exclude black pixels
    Mat mask;
    inRange(img, Scalar(0, 0, 0), Scalar(0, 0, 1), mask); // Only mask black pixels
    bitwise_not(mask, mask); // Invert mask to include non-black pixels

    // Compute the histogram
    Mat hist;
    calcHist(&img, 1, channels, mask, hist, 3, hist_size, ranges);

    // Find the bin with the maximum count
    double max_val = 0;
    int max_idx[3] = {0, 0, 0};
    minMaxIdx(hist, nullptr, &max_val, nullptr, max_idx);

    // Convert the bin index to HSV color
    int h_bin = max_idx[0], s_bin = max_idx[1], v_bin = max_idx[2];
    float h_step = 180.0f / h_bins, s_step = 256.0f / s_bins, v_step = 256.0f / v_bins;
    most_common_color = Vec3b(static_cast<uchar>(h_bin * h_step),static_cast<uchar>(s_bin * s_step),static_cast<uchar>(v_bin * v_step));
}

void contrastStretching(const Mat& img, Mat& dest)  {
    // HSV conversion
    Mat hsv_img;
    cvtColor(img,hsv_img,COLOR_BGR2HSV_FULL);

    // Split the HSV image into channels
    std::vector<Mat> hsvChannels;
    split(hsv_img,hsvChannels);

    // Compute the standard deviation of the image as measure of contrast
    Scalar mean, stddev;
    meanStdDev(hsvChannels[2],mean,stddev);

    // Define the brightness increase as function of the standard deviation
    int brightness_increase = static_cast<int>(0.5 * stddev[0]);

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
    merge(hsvChannels,hsv_img);

    cvtColor(hsv_img,dest,COLOR_HSV2BGR_FULL);
}

void adaptiveColorBasedSegmentation(const Mat& img, Mat& dest, std::vector<int> HSV_thresholds, double window_ratio) {
    // Convert the image to HSV color space
    Mat hsv_img;
    cvtColor(img, hsv_img, COLOR_BGR2HSV_FULL);

    // Calculate the window size as function of the image size
    int window_size = static_cast<int>(std::round(static_cast<double>(std::max(img.rows,img.cols)) / window_ratio));

    // Initialize the destination image
    dest = Mat::zeros(img.size(), img.type());

    // Iterate over the image with non-overlapping windows
    for (int y = 0; y < img.rows; y += window_size) {
        for (int x = 0; x < img.cols; x += window_size) {
            // Define the window region, ensuring it doesn't exceed the image bounds
            int window_width = std::min(window_size, img.cols - x);
            int window_height = std::min(window_size, img.rows - y);
            Rect window(x, y, window_width, window_height);
            Mat window_region = hsv_img(window);

            // Determine the most common color in the window region by histogram evaluation
            Vec3b most_common_color;
            mostCommonColor(window_region,most_common_color);

            // Create a mask for the most common color in the window region
            Scalar lower_bound(most_common_color[0] - HSV_thresholds[0], most_common_color[1] - HSV_thresholds[1], most_common_color[2] - HSV_thresholds[2]);
            Scalar upper_bound(most_common_color[0] + HSV_thresholds[0], most_common_color[1] + HSV_thresholds[1], most_common_color[2] + HSV_thresholds[2]);
            Mat mask;
            inRange(window_region, lower_bound, upper_bound, mask);

            // Invert the mask to remove the most common color
            Mat inverted_mask;
            bitwise_not(mask,inverted_mask);

            // Apply the mask to the corresponding region in the destination image
            Mat dest_region = dest(window);
            img(window).copyTo(dest_region,inverted_mask);
        }
    }
}

void ballSelection(const Mat& img, const std::vector<Vec3f>& circle_vector, std::vector<Vec3f>& new_circle_vector, const std::vector<billiardSet> billiard_tables) {
    Mat mask, ball_region, hsv_ball_region;
    Vec3b dominant_color;
    Vec3b diff = {0,0,0};
    bool is_white,is_black,is_yellow,is_blue,is_red,is_purple,is_orange,is_green,is_brown;

    // Iterate for each circle
    for (int i = 0; i < circle_vector.size(); i++) {
        // Define circle center and radius
        Point center(cvRound(circle_vector[i][0]), cvRound(circle_vector[i][1]));
        int radius = cvRound(circle_vector[i][2]);

        // Compute the circle mask and apply it to img
        mask = Mat::zeros(img.size(),CV_8U);
        ball_region = Mat::zeros(img.size(),CV_8U);
        circle(mask,center,radius,Scalar(255),-1);
        Mat eroded_mask;
        erode(mask,eroded_mask, getStructuringElement(MORPH_ELLIPSE,Size(3,3)),Point(-1,-1),4);
        img.copyTo(ball_region,eroded_mask);

        // Convert the ball region in HSV color-space
        cvtColor(ball_region,hsv_ball_region, COLOR_BGR2HSV_FULL);

        // Compute the dominant color
        mostCommonColor(hsv_ball_region,dominant_color);

        // Dilate the mask if the detected color is invalid
        if (dominant_color == Vec3b(0,0,0)) {
            Mat dilated_mask;
            dilate(eroded_mask,dilated_mask,getStructuringElement(MORPH_ELLIPSE,Size(3,3)));
            img.copyTo(ball_region,dilated_mask);
            cvtColor(ball_region,hsv_ball_region, COLOR_BGR2HSV_FULL);
            mostCommonColor(hsv_ball_region,dominant_color);
        }

        // verify that the dominant color belongs to one of the billiard ball colors
        for (const auto & billiard_table : billiard_tables) {
            for (const auto & ball : billiard_table.billiard_set) {
                // Compute the absolute difference between dominant and ball color
                absdiff(dominant_color,ball.color_value,diff);

                // Check the color ranges
                is_white = (ball.color_name == "White") && (diff[0] < 4) && (diff[1] < 9) && (diff[2] < 2);
                is_black = (ball.color_name == "Black") && (diff[0] < 2) && (diff[1] < 2) && (diff[2] < 2);
                is_yellow = (ball.color_name == "Yellow") && (diff[0] < 4) && (diff[1] < 25) && (diff[2] < 2);
                is_blue = (ball.color_name == "Blue") && (diff[0] < 2) && (diff[1] < 2) && (diff[2] < 2);
                is_red = (ball.color_name == "Red") && (diff[0] < 2) && (diff[1] < 17) && (diff[2] < 13);
                is_purple = (ball.color_name == "Purple") && (diff[0] < 4) && (diff[1] < 25) && (diff[2] < 5);
                is_orange = (ball.color_name == "Orange") && (diff[0] < 4) && (diff[1] < 17) && (diff[2] < 13);
                is_green = (ball.color_name == "Green") && (diff[0] < 2) && (diff[1] < 2) && (diff[2] < 2);
                is_brown = (ball.color_name == "Brown") && (diff[0] < 2) && (diff[1] < 2) && (diff[2] < 9);

                // If dominant color is accepted, it is appended to the new circle vector
                if (is_white || is_black || is_yellow || is_blue || is_red || is_purple || is_orange || is_green || is_brown) {
                    new_circle_vector.emplace_back(circle_vector[i]);
                }
                diff = {0,0,0};
            }
        }
    }
}

void ballDetection(const Mat& img, std::vector<Vec3f> circles) {
    // Bilateral Filter [d:7, sigmaColor:60, sigmaSpace:300]
    Mat filtered_img;
    bilateralFilter(img,filtered_img,7,60,300);

    // Contrast stretching used to enhance dark regions, and hence obtain a correct segmentation
    contrastStretching(filtered_img,filtered_img);

    // Color-based segmentation applied to obtain the balls mask
    Mat segmented_img;
    double window_ratio = 14.6;
    std::vector<int> HSV_thresholds = {8, 67, 67};
    adaptiveColorBasedSegmentation(filtered_img,segmented_img,HSV_thresholds,window_ratio);

    // Conversion to gray-scale and binary thresholding of the balls mask
    cvtColor(segmented_img,segmented_img,COLOR_BGR2GRAY);
    Mat binary_segmented_img;
    threshold(segmented_img,binary_segmented_img,0,255,THRESH_BINARY);

    // Morphological operators (CLOSING + OPENING), used to make more even the balls blobs
    morphologyEx(binary_segmented_img,binary_segmented_img,MORPH_CLOSE,getStructuringElement(MORPH_ELLIPSE,Size(3, 3)),
                 Point(-1, -1),1);
    morphologyEx(binary_segmented_img,binary_segmented_img,MORPH_OPEN,getStructuringElement(MORPH_ELLIPSE,Size(3,3)),
                 Point(-1,-1),3);

    // Hough circles transformation for circle detection on the binary mask
    double min_distance_between_circles = static_cast<double>(binary_segmented_img.cols) / 40;
    int thresh1 = 300;
    int thresh2 = 6;
    double min_radius = static_cast<double>(std::max(binary_segmented_img.cols, binary_segmented_img.rows)) / 115;
    double max_radius = static_cast<double>(std::max(binary_segmented_img.cols, binary_segmented_img.rows)) / 35;
    std::vector<Vec3f> detected_circles;
    HoughCircles(binary_segmented_img,detected_circles,HOUGH_GRADIENT,1,min_distance_between_circles,thresh1,thresh2,
                 min_radius, max_radius);

    // Dataset of the ball colors for each table
    std::vector<billiardSet> billiard_tables = {
            {"Palette_1",{
                                 {"White", Vec3b(45,68,248)},
                                 {"Black", Vec3b(144,224,40)},
                                 {"Yellow", Vec3b(18,200,168)},
                                 {"Blue", Vec3b(78,56,160)},
                                 {"Red", Vec3b(138,160,128)},
                                 {"Purple", Vec3b(162, 136, 88)},
                                 {"Orange", Vec3b(0, 192, 152)},
                                 {"Green", Vec3b(114, 224, 72)},
                                 {"Brown", Vec3b(12, 168, 112)}
                         }},
            {"Palette_2",{
                                 {"White", Vec3b(42,8,232)},
                                 {"Black", Vec3b(132,248,64)},
                                 {"Yellow", Vec3b(12,96,248)},
                                 {"Blue", Vec3b(150,184,136)},
                                 {"Red", Vec3b(60,32,232)},
                                 {"Purple", Vec3b(156,72,104)},
                                 {"Orange", Vec3b(21,80,248)},
                                 {"Green", Vec3b(102,168,88)},
                                 {"Brown", Vec3b(0, 56,160)}
                         }},
            {"Palette_3",{
                                 {"White", Vec3b(84,8,240)},
                                 {"Black", Vec3b(132,128,48)},
                                 {"Yellow", Vec3b(36,176,248)},
                                 {"Blue", Vec3b(150,192,200)},
                                 {"Red", Vec3b(0,16,240)},
                                 {"Purple", Vec3b(168,136,136)},
                                 {"Orange", Vec3b(18,160,244)},
                                 {"Green", Vec3b(102,104,176)},
                                 {"Brown", Vec3b(168,32,48)}
                         }},
            {"Palette_4",{
                                 {"White", Vec3b(33,80,248)},
                                 {"Black", Vec3b(6,32,56)},
                                 {"Yellow", Vec3b(27,216,248)},
                                 {"Blue", Vec3b(36,40,248)},
                                 {"Red", Vec3b(138,88,128)},
                                 {"Purple", Vec3b(6,16,232)},
                                 {"Orange", Vec3b(6,224,232)},
                                 {"Green", Vec3b(96,112,128)},
                                 {"Brown", Vec3b(12,176,192)}
                         }},
            {"Palette_5",{
                                 {"White", Vec3b(42,60,248)},
                                 {"Black", Vec3b(138,248,48)},
                                 {"Yellow", Vec3b(42,80,248)},
                                 {"Blue", Vec3b(156,192,88)},
                                 {"Red", Vec3b(138,112,80)},
                                 {"Purple", Vec3b(168,120,88)},
                                 {"Orange", Vec3b(42,104,208)},
                                 {"Green", Vec3b(114,232,72)},
                                 {"Brown", Vec3b(54,16,240)}
                         }},
            {"Palette_6",{
                                 {"White", Vec3b(48,88,200)},
                                 {"Black", Vec3b(144,224,40)},
                                 {"Yellow", Vec3b(18,184,152)},
                                 {"Blue", Vec3b(126,144,88)},
                                 {"Red", Vec3b(0, 152,240)},
                                 {"Purple", Vec3b(165,120,84)},
                                 {"Orange", Vec3b(6,204,212)},
                                 {"Green", Vec3b(126,232,56)},
                                 {"Brown", Vec3b(54,56,160)}
                         }},
            {"Palette_7",{
                                 {"White", Vec3b(42,16,248)},
                                 {"Black", Vec3b(114,248,32)},
                                 {"Yellow", Vec3b(36,64,224)},
                                 {"Blue", Vec3b(126,248,56)},
                                 {"Red", Vec3b(6,56,120)},
                                 {"Purple", Vec3b(162,160,80)},
                                 {"Orange", Vec3b(6,184,104)},
                                 {"Green", Vec3b(108,248,88)},
                                 {"Brown", Vec3b(18,24,224)}
                         }},
            {"Palette_8",{
                                 {"White", Vec3b(42,40,248)},
                                 {"Black", Vec3b(114,224,64)},
                                 {"Yellow", Vec3b(24,192,232)},
                                 {"Blue", Vec3b(0,104,216)},
                                 {"Red", Vec3b(12,56,248)},
                                 {"Purple", Vec3b(168,128,80)},
                                 {"Orange", Vec3b(6,200,248)},
                                 {"Green", Vec3b(108,248,64)},
                                 {"Brown", Vec3b(42,0,248)}
                         }},
            {"Palette_9",{
                                 {"White", Vec3b(36,64,232)},
                                 {"Black", Vec3b(174,104,32)},
                                 {"Yellow", Vec3b(24,232,232)},
                                 {"Blue", Vec3b(144,128,160)},
                                 {"Red", Vec3b(0,224,144)},
                                 {"Purple", Vec3b(0,160,236)},
                                 {"Orange", Vec3b(6,232,216)},
                                 {"Green", Vec3b(96,136,120)},
                                 {"Brown", Vec3b(12,128,216)}
                         }},
            {"Palette_10",{
                                 {"White", Vec3b(36,48,248)},
                                 {"Black", Vec3b(72,128,16)},
                                 {"Yellow", Vec3b(24,160,152)},
                                 {"Blue", Vec3b(150,224,96)},
                                 {"Red", Vec3b(0,196,200)},
                                 {"Purple", Vec3b(12,120,248)},
                                 {"Orange", Vec3b(12,176,248)},
                                 {"Green", Vec3b(102,184,56)},
                                 {"Brown", Vec3b(12,176,136)}
                         }}
    };

    // Color-based ball selection and rejection of false positive samples   [for reference -> paper "3D reconstruction..."]
    ballSelection(img,detected_circles,circles,billiard_tables);
}

void drawCircles(const Mat& img, Mat& circles_img, const std::vector<Vec3f>& circles) {
    img.copyTo(circles_img);
    // Visualize the detected balls in the original image
    for (size_t i = 0; i < circles.size(); i++) {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        // draw the circle center
        circle(circles_img,center,1,Scalar(0, 255, 0),1, LINE_AA);
        // draw the circle outline
        circle(circles_img,center,radius,Scalar(0, 0, 255),1, LINE_AA);
    }
}

void printCircles(const Mat& img, const std::vector<Vec3f> circles, std::vector<Mat> circles_img) {
    Mat mask, ball_region;
    for (int i = 0; i < circles.size(); ++i) {
        // Define circle center and radius
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);

        // Compute the circle mask and apply it to img
        mask = Mat::zeros(img.size(),CV_8U);
        ball_region = Mat::zeros(img.size(),CV_8U);
        circle(mask,center,radius,Scalar(255),-1);
        img.copyTo(ball_region,mask);

        // Convert the image to grayscale
        Mat gray_ball_region;
        cvtColor(ball_region,gray_ball_region,COLOR_BGR2GRAY);

        // Threshold the grayscale image to get a binary mask
        Mat binary_ball_region;
        threshold(gray_ball_region,binary_ball_region,1,255,THRESH_BINARY);

        // Find the bounding box of the non-black region (subject)
        Rect boundingBox = boundingRect(binary_ball_region);

        // Crop the image using the bounding box coordinates
        Mat cropped_ball_region = ball_region(boundingBox);

        circles_img.emplace_back(cropped_ball_region);
    }
}