#include "ball_detection.h"
#include "classification.h"
using namespace cv;
using namespace std;

void printCircles(const Mat& img, const std::vector<billiardBall>& balls, std::vector<Mat>& circles_img) {
    Mat mask, ball_region;
    for (size_t i = 0; i < balls.size(); ++i) {
        // Define circle center and radius
        Point center(cvRound(balls[i].x), cvRound(balls[i].y));
        int radius = cvRound(balls[i].true_radius);

        // Compute the circle mask and apply it to img
        mask = Mat::zeros(img.size(), CV_8U);
        ball_region = Mat::zeros(img.size(), img.type()); // Ensure ball_region has the same type as img
        circle(mask, center, radius, Scalar(255), -1);
        erode(mask,mask, getStructuringElement(MORPH_ELLIPSE,Size(3,3)),
              Point(-1,-1),1);
        img.copyTo(ball_region, mask);

        Mat binary_ball_region;
        cvtColor(ball_region,binary_ball_region,COLOR_BGR2GRAY);
        threshold(binary_ball_region,binary_ball_region,1,255,THRESH_BINARY);
        Rect boundingBox = boundingRect(binary_ball_region);

        Mat cropped_ball_region = ball_region(boundingBox).clone();

        /*
        int dim;
        // if white 0.56 size 8
        if (white_percentage >0.5){
            dim = 8;
        }
        else{
            // If more white pixels, larger size of rect
            dim = static_cast<int>(30*white_percentage);
        }
        cv::Rect rect(cropped_ball_region.rows/2 - dim/2, cropped_ball_region.cols/2 - dim/2, dim, dim);

        // Crop the image using the rectangle
        Mat center_image;
        center_image = cropped_ball_region(rect).clone();
        // Create a new image with added borders

        int borderSize = 10;
        Mat augmented;
        copyMakeBorder(center_image, augmented, borderSize, borderSize, borderSize, borderSize, BORDER_REPLICATE);
        // Resize the image
        //resize(augmented, show, newSize);
        //imshow("EROSION",show);
        //waitKey();
        */
        // Add the new image to the circles_img vector
        circles_img.push_back(cropped_ball_region);
    }
}

void mostCommonColorForClassification(const Mat& img, Vec3b& most_common_color, double& count) {
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
    erode(mask,mask, getStructuringElement(MORPH_ELLIPSE,Size(3,3)),Point(-1,-1),1);

    // Compute the histogram
    Mat hist;
    calcHist(&img, 1, channels, mask, hist, 3, hist_size, ranges);
    // Find the bin with the maximum count
    int max_idx[3] = {0, 0, 0};
    minMaxIdx(hist, nullptr, &count, nullptr, max_idx);

    // Convert the bin index to HSV color
    int h_bin = max_idx[0], s_bin = max_idx[1], v_bin = max_idx[2];
    float h_step = 180.0f / h_bins, s_step = 256.0f / s_bins, v_step = 256.0f / v_bins;
    most_common_color = Vec3b(static_cast<uchar>(h_bin * h_step),static_cast<uchar>(s_bin * s_step),static_cast<uchar>(v_bin * v_step));
}

cv::Mat extractTextureFeatures(const cv::Mat& img) {
    cv::Mat imgGray, hist;
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    // Example: Compute texture features (e.g., Local Binary Patterns)
    // This is a placeholder for actual texture feature extraction
    // For simplicity, we'll use the same histogram computation here

    int textureBins = 16; // Number of bins for texture histogram
    int histSize[] = { textureBins };
    float textureRange[] = { 0, 256 };
    const float* ranges[] = { textureRange };
    int channels[] = { 0 };
    cv::MatND hist_texture;

    cv::calcHist(&imgGray, 1, channels, cv::Mat(), hist_texture, 1, histSize, ranges, true, false);
    cv::normalize(hist_texture, hist_texture, 0, 1, cv::NORM_MINMAX);

    // Flatten histogram to 1D feature vector
    cv::Mat textureFeatureVector = hist_texture.reshape(1, 1);

    return textureFeatureVector;
}
cv::Mat extractColorFeatures(const cv::Mat& img) {
    cv::Mat imgGray,  hist_texture;

    // Convert image to grayscale
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    // Compute color histogram in grayscale space
    int hBins = 256; // Number of bins for grayscale histogram
    int histSize[] = { hBins };
    float hRange[] = { 0, 256 };
    const float* ranges[] = { hRange };
    int channels[] = { 0 };
    cv::MatND hist_gray;
    cv::calcHist(&imgGray, 1, channels, cv::Mat(), hist_gray, 1, histSize, ranges, true, false);
    cv::normalize(hist_gray, hist_gray, 0, 1, cv::NORM_MINMAX);

    // Flatten grayscale histogram to 1D feature vector
    cv::Mat grayFeatureVector = hist_gray.reshape(1, 1);

    // Compute additional texture features
    cv::Mat textureFeatureVector = extractTextureFeatures(img);

    // Concatenate grayscale and texture features horizontally
    std::vector<cv::Mat> features;
    features.push_back(grayFeatureVector);
    features.push_back(textureFeatureVector);

    cv::Mat combinedFeatureVector;
    cv::hconcat(features, combinedFeatureVector);

    return combinedFeatureVector;
}

void classification(const cv::Mat& img, std::vector<billiardBall>& balls) {

    vector<Mat> circles_img;
    printCircles(img,balls,circles_img);

    // Thresholds on the count of the most common color
    int thresh_white_count = 12;
    int thresh_black_count = 15;
    // Vectors for storing the whitest and the darkest balls on the table
    std::vector<double> white_counts;
    std::vector<int> white_indices;
    std::vector<double> black_counts;
    std::vector<int> black_indices;

    // Iterate over all the balls
    for (int i = 0; i < balls.size(); i++) {
        Mat filtered_image;
        bilateralFilter(circles_img[i],filtered_image,5,60,300);

        // Compute the most common color on the HSV version of the ball
        Mat hsv_ball;
        cvtColor(filtered_image, hsv_ball, COLOR_BGR2HSV_FULL);
        Vec3b most_common_color;
        double count;
        // Evaluate the most common color on the eroded ball mask, in order to avoid
        // considering the points of the contours
        mostCommonColorForClassification(hsv_ball,most_common_color,count);

        if (most_common_color[0] >= 30 && most_common_color[0] <= 50 && most_common_color[2] >= 200 && count >= thresh_white_count) {
            white_counts.push_back(count);
            white_indices.push_back(i);
        } else if (most_common_color[2] <= 65 && count >= thresh_black_count) {
            black_counts.push_back(count);
            black_indices.push_back(i);
        }
    }

    // CLASSIFY WHITE BALLS
    if (!white_counts.empty()) {
        auto max_white_iterator = std::max_element(white_counts.begin(),white_counts.end());
        int max_white_index = std::distance(white_counts.begin(),max_white_iterator);

        balls[white_indices[max_white_index]].id = 0;
    }

    if (!black_counts.empty()) {
        auto max_black_iterator = std::max_element(black_counts.begin(),black_counts.end());
        int max_black_index = std::distance(black_counts.begin(),max_black_iterator);

        balls[black_indices[max_black_index]].id = 1;
    }

    int i = 0, j = 0;
    for (const auto& ball: balls) {
        if (ball.id == 0)
            i++;
        if(ball.id == 1)
            j++;
    }
    cout << "COUNT  " << i << "  " << j << endl;

    // SOLID/STRIPES CLASSIFICATION
    std::vector<cv::Mat> features;

    // Read images and compute histograms
    for (size_t i = 0; i < circles_img.size(); ++i) {
        if(balls[i].id!=0 || balls[i].id!=1){
            cv::Mat feature = extractColorFeatures(circles_img[i]);
            features.push_back(feature);
        }

    }

    // Convert vector of features to a single matrix
    cv::Mat featureMatrix;
    cv::vconcat(features, featureMatrix);

    // Apply KMeans clustering
    int clusterCount = 2;
    cv::Mat labels;
    cv::Mat centers;
    theRNG().state = 12345;
    cv::kmeans(featureMatrix, clusterCount, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 0.2),
               1000, cv::KMEANS_PP_CENTERS, centers);

    // Output clustering results
    for (size_t i = 0; i < balls.size(); ++i) {
        if(balls[i].id!=0 && balls[i].id!=1){
            if(labels.at<int>(i)==0){
                balls[i].id = 2;
            }
            else{
                balls[i].id = 3;
            }
            std::cout << "Image " << i << " belongs to cluster " << labels.at<int>(i) << std::endl;
        }

    }

    // VISUALIZATION
    // VISUALIZE DETECTED BALLS
    Mat detected_balls = img.clone();
    for (int i = 0; i < balls.size(); i++) {
        Point center(balls[i].x, balls[i].y);
        Scalar color;
        if(balls[i].id==0){
            color = Scalar(255, 255, 255); // White
        }
        if(balls[i].id==1){
            color = Scalar(0, 0, 0); // Black
        }
        if(balls[i].id==2){
            color = Scalar(0, 255, 255); // Yellow
        }
        if(balls[i].id==3){
            color = Scalar(0, 255, 0); // Green
        }

        circle(detected_balls, center, balls[i].true_radius, color, 2, LINE_AA);
    }
}

