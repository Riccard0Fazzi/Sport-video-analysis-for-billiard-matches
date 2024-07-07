// Created by Tommaso Tubaldo on 06/06/24 - Hours: 20
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

using namespace cv;

void contrastStretching(const Mat& img, Mat& dest);
void adaptiveColorBasedSegmentation(const Mat& img, Mat& dest, std::vector<int> HSV_thresholds, double window_ratio);

int main(int argc, char** argv) {
    if (argc < 3) {
        std :: cout << "An image file and a image name with folder path should be provided!";
        return EXIT_FAILURE;
    }   // Checks for the correctness of the input values

    std::string path = argv[1];
    std::string pattern = argv[3];
    std::vector<cv::String> filenames;
    utils::fs::glob(path,pattern,filenames);

    // Loop for every image     (size_t c = 0; c < filenames.size(); ++c)
    for (size_t c = 6; c < 7; ++c) {
        std::string num = std::to_string(c);

        // Read the image
        Mat img = imread(filenames[c], IMREAD_ANYCOLOR);
        // Safety check on the image returned
        if (img.empty()) // If filename is wrong, imread returns an empty Mat object
        {
            // Print an error message using cv::Error
            std::cerr << "Error: " << format("Failed to load image! Error code: %d", cv::Error::StsError) << std::endl;
            exit(0);
        }

        // Bilateral Filter [d:7, sigmaColor:80, sigmaSpace:300]
        Mat filtered_img;
        bilateralFilter(img, filtered_img, 7, 60, 300);
        //std::string output_img_name13 = "/bilateralFilter.png";
        //imwrite(argv[2] + output_img_name13, filtered_img);

        // Contrast stretching used to enhance dark regions, and hence obtain a correct segmentation
        int brightness_increase = 60;
        contrastStretching(filtered_img,filtered_img);
        std::string output_img_name16 = "/equalized_HSV";
        imwrite(argv[2] + output_img_name16 + num + ".png", filtered_img);

        // Color-based segmentation applied to obtain the balls mask
        Mat segmented_img;
        double window_ratio = 14.6;   // last used -> 14.6
        std::vector<int> HSV_thresholds = {8, 40, 80};   // last used -> [8,80,80]
        adaptiveColorBasedSegmentation(filtered_img,segmented_img,HSV_thresholds,window_ratio);
        std::string output_img_name8 = "/mask";
        imwrite(argv[2] + output_img_name8 + num + ".png", segmented_img);

        // Conversion to gray-scale and binary thresholding of the balls mask
        cvtColor(segmented_img,segmented_img,COLOR_BGR2GRAY);
        Mat binary_segmented_img;
        threshold(segmented_img,binary_segmented_img,0,255,THRESH_BINARY);
        std::string output_img_name9 = "/mask_binary.png";
        imwrite(argv[2] + output_img_name9, binary_segmented_img);


        // Morphological operators (CLOSING + OPENING) used to make more even the balls blobs
        morphologyEx(binary_segmented_img,binary_segmented_img,MORPH_CLOSE,getStructuringElement(MORPH_ELLIPSE,Size(3, 3)),
                     Point(-1, -1),1);
        morphologyEx(binary_segmented_img,binary_segmented_img,MORPH_OPEN,getStructuringElement(MORPH_ELLIPSE,Size(3,3)),
                     Point(-1,-1),3);
        std::string output_img_name3 = "/opened_mask";
        imwrite(argv[2] + output_img_name3 + num + ".png", binary_segmented_img);

        // CIRCLE DETECTION from the binary mask
        // Hough circles transformation
        double min_distance_between_circles = static_cast<double>(binary_segmented_img.cols) / 40;
        int thresh1 = 300;
        int thresh2 = 9;
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
            std::cout << "No circles detected!!\n";
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

        std::string output_img_name2 = "/circles";
        imwrite(argv[2] + output_img_name2 + num + ".png", img);  // Saving the image on the desired folder

    }
    return 0;
}

void contrastStretching(const Mat& img, Mat& dest) {
    // Converts img to grayscale
    Mat gray;
    cvtColor(img,gray,COLOR_BGR2GRAY);

    // Compute the standard deviation of the image as measure of contrast
    Scalar mean, stddev;
    meanStdDev(gray,mean,stddev);

    // Define the brightness increase as function of the standard deviation
    int brightness_increase = static_cast<int>(0.9 * stddev[0]);
    std::cout << brightness_increase << "\n";

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

            // Calculate the histogram for the current window region
            int h_bins = 30, s_bins = 32, v_bins = 32;
            int hist_size[] = {h_bins, s_bins, v_bins};

            float h_range[] = {0, 180};
            float s_range[] = {0, 256};
            float v_range[] = {0, 256};
            const float* ranges[] = {h_range, s_range, v_range};

            int channels[] = {0, 1, 2};

            Mat hist;
            calcHist(&window_region, 1, channels, Mat(), hist, 3, hist_size, ranges);
            hist.at<float>(0) = 0;

            // Find the bin with the maximum count
            double max_val = 0;
            int max_idx[3] = {0, 0, 0};
            minMaxIdx(hist, nullptr, &max_val, nullptr, max_idx);

            // Convert the bin index to HSV color
            int h_bin = max_idx[0], s_bin = max_idx[1], v_bin = max_idx[2];
            float h_step = 180.0f / h_bins, s_step = 256.0f / s_bins, v_step = 256.0f / v_bins;
            Vec3b most_common_color(h_bin * h_step, s_bin * s_step, v_bin * v_step);

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