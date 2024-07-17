// Created by Tommaso Tubaldo on 06/06/24 - Hours: 70
#include "ball_detection.hpp"
using namespace cv;
using namespace std;

// Structure used to store balls names and colors
billiardBall::billiardBall(int x, int y, int width, int height, cv::Mat& image)
    : x(x), y(y), width(width), height(height), image(image)
{
    // Optionally, you can add additional initialization logic here if needed
}

std::vector<billiardBall> ball_detection(const cv::Mat& inputImage)
{

	Mat img = inputImage.clone();

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

	std::vector<Mat> circles_images;
	int circle_size = 100;
	printCircles(img,circles,circle_size,circles_images);
	
	// Draw the detected circles
	Mat circles_img;
	drawCircles(img,circles_img,circles);
	std::vector<billiardBall> balls; // vector of object balls
	
	for(int i = 0; i < circles_images.size(); i++)
	{
		balls.emplace_back(circles[i][0],circles[i][1],circles[i][2],10,circles_images[i]);		
	}
	

	imshow("Circles",circles_img);
	waitKey(0);
	destroyAllWindows();

	return balls;
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
void adaptiveColorBasedSegmentation(const Mat& img, Mat& dest, double window_ratio) {

	Mat temp = img.clone();
	cvtColor(temp,temp,COLOR_BGR2Lab);
	pyrMeanShiftFiltering(temp,temp,10,15);
	cvtColor(temp,temp,COLOR_Lab2BGR);
	namedWindow("MeanShift");
	imshow("MeanShift",temp);
	waitKey(0);

    // Convert the image to HSV color space
    Mat hsv_img;
    cvtColor(temp, hsv_img, COLOR_BGR2HSV_FULL);
    std::vector<Mat> img_channels;
    split(hsv_img, img_channels);



    // Compute the mean and standard deviation for each channel
    Scalar mh, sh;
    meanStdDev(img_channels[0], mh, sh);


    Scalar ms, ss;
    meanStdDev(img_channels[1], ms, ss);

    Scalar mv, sv;
    meanStdDev(img_channels[2], mv, sv);

    cout << "mh: " << mh[0] << "ms: " << ms[0] << "mv: " << mv[0] << endl;

    // Calculate the window size as function of the image size
    int window_size = static_cast<int>(std::round(static_cast<double>(std::max(img.rows,img.cols)) / window_ratio));



    // Initialize the destination imageq
    dest = Mat::zeros(img.size(), img.type());

    // Determine the most common color in the window region by histogram evaluation
    Vec3b field_color;
    mostCommonColor(img,field_color);
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




            std::vector<Mat> hsv_channels;
            split(window_region, hsv_channels);


            // Compute the mean and standard deviation for each channel
            Scalar mean_hue, stddev_hue;
            meanStdDev(hsv_channels[0], mean_hue, stddev_hue);
            Scalar nmh = mean_hue/mh;
            Scalar nsh = stddev_hue/sh;


            Scalar mean_saturation, stddev_saturation;
            meanStdDev(hsv_channels[1], mean_saturation, stddev_saturation);
            Scalar nms = mean_saturation/ms;
            Scalar nss = stddev_saturation/ss;

            Scalar mean_value, stddev_value;
            meanStdDev(hsv_channels[2], mean_value, stddev_value);
            Scalar nmv = mean_value/mv;
            Scalar nsv = stddev_value/sv;


            // Set thresholds based on mean and stddev

            // Calculate lower and upper bounds directly based on standard deviations
            std::vector<int> hsv_thresholds(6);

            // GENERAL STD variation
            double lower_coeff = 0.9; // 0.8
            double higher_coeff = 1.2; // 1.2

            double lower_weight[3];
            double higher_weight[3];

			// Hue settings for lower/upper bound differences
			if(most_common_color[0] > mean_hue[0])
			{
				higher_weight[0] = 0.8;
				lower_weight[0]= 1.2;
			} else
			{
				higher_weight[0] = 1.2;
				lower_weight[0] = 0.8;
			}
			if(most_common_color[1] > mean_hue[1])
			{
				higher_weight[1] = 0.8;
				lower_weight[1] = 1.2;
			} else
			{
				higher_weight[1] = 1.2;
				lower_weight[1] = 0.8;
			}
			if(most_common_color[2] > mean_hue[2])
			{
				higher_weight[2] = 0.8;
				lower_weight[2] = 1.2;
			} else
			{
				higher_weight[2] = 1.2;
				lower_weight[2] = 0.8;
			}
            // HUE

            // STD condition
            double h_cond;
            if(stddev_hue[0]>sh[0]){
                h_cond = sh[0]/stddev_hue[0];
            }
            else{
                h_cond = stddev_hue[0]/sh[0];
            }

            // THRESHOLD
            double  h_t;
            if(mean_hue[0]>mh[0]){
                h_t = 10*mh[0]/mean_hue[0];
            }
            else{
                h_t = 10*mean_hue[0]/mh[0];
            }

                     // 0.3 && h_t/10 < 0.5)
            if(h_cond < 0.5) // case 1: non-uniform window only in Hue
            {
                hsv_thresholds[0] = static_cast<int>(lower_weight[0]*lower_coeff*h_t); // Lower bound for hue
                hsv_thresholds[1] = static_cast<int>(higher_weight[0]*lower_coeff*h_t); // Upper bound for hue

            } else {

                hsv_thresholds[0] = static_cast<int>(lower_weight[0]*higher_coeff*h_t); // Lower bound for hue
                hsv_thresholds[1] = static_cast<int>(higher_weight[0]*higher_coeff*h_t);  // Upper bound for hue
            }

            // SATURATION

            // STD condition
            double s_cond;
            if(stddev_saturation[0]>ss[0]){
                s_cond = ss[0]/stddev_saturation[0];
            }
            else{
                s_cond = stddev_saturation[0]/ss[0];
            }

            // THRESHOLD
            double  s_t;
            if(mean_saturation[0]>ms[0]){
                s_t = 60*ms[0]/mean_saturation[0];
            }
            else{
                s_t = 60*mean_saturation[0]/ms[0];
            }

			       // 0.3
            if(s_cond<0.6)
            {
                hsv_thresholds[2] = static_cast<int>(lower_weight[1]*lower_coeff*s_t); // Lower bound for saturation
                hsv_thresholds[3] = static_cast<int>(higher_weight[1]*lower_coeff*s_t); // Upper bound for saturation


            } else {

                hsv_thresholds[2] = static_cast<int>(lower_weight[1]*higher_coeff*s_t);  // Lower bound for saturation
                hsv_thresholds[3] = static_cast<int>(higher_weight[1]*higher_coeff*s_t);  // Upper bound for saturation
            }



            //VALUE

            // STD condition
            double v_cond;
            if(stddev_value[0]>sv[0]){
                v_cond = sv[0]/stddev_value[0];
            }
            else{
                v_cond = stddev_value[0]/sv[0];
            }
            // THRESHOLD
            double  v_t;
            if(mean_value[0]>mv[0]){
                v_t = 60*mv[0]/mean_value[0]; // 60
            }
            else{
                v_t = 60*mean_value[0]/mv[0]; // 60
            }
				   // 0.6
            if(v_cond<0.6)
            {
                hsv_thresholds[4] = static_cast<int>(lower_weight[2]*lower_coeff*v_t); // Lower bound for value
                hsv_thresholds[5] = static_cast<int>(higher_weight[2]*lower_coeff*v_t); // Upper bound for value


            } else {

                hsv_thresholds[4] = static_cast<int>(lower_weight[2]*higher_coeff*v_t);  // Lower bound for value
                hsv_thresholds[5] = static_cast<int>(higher_weight[2]*higher_coeff*v_t);  // Upper bound for value
            }
            // Create a mask for the most common color in the window region
            Scalar lower_bound(
                    std::max(most_common_color[0] - hsv_thresholds[0], 0),
                    std::max(most_common_color[1] - hsv_thresholds[2], 0),
                    std::max(most_common_color[2] - hsv_thresholds[4], 0)
            );
            Scalar upper_bound(
                    std::min(most_common_color[0] + hsv_thresholds[1], 180),
                    std::min(most_common_color[1] + hsv_thresholds[3], 255),
                    std::min(most_common_color[2] + hsv_thresholds[5], 255)
            );
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


void ballDetection(const Mat& img, std::vector<Vec3f>& circles) {
    // Bilateral Filter [d:7, sigmaColor:60, sigmaSpace:300]
    Mat filtered_img;
    bilateralFilter(img,filtered_img,7,60,300);

    // Contrast stretching used to enhance dark regions, and hence obtain a correct segmentation
    contrastStretching(filtered_img,filtered_img);

    // Color-based segmentation applied to obtain the balls mask
    Mat segmented_img;
    double window_ratio = 11.5; //14.6;
    //std::vector<int> HSV_thresholds = {8, 200, 70};
    adaptiveColorBasedSegmentation(filtered_img,segmented_img,window_ratio);

	// Here the image is converted into a binary one
	// ideally only the billiard balls should be highlighted
    // Conversion to gray-scale and binary thresholding of the balls mask
    cvtColor(segmented_img,segmented_img,COLOR_BGR2GRAY);
    Mat binary_segmented_img;
    threshold(segmented_img,binary_segmented_img,0,255,THRESH_BINARY);

    // Morphological operators (CLOSING + OPENING), used to make more even the balls blobs
    morphologyEx(binary_segmented_img,binary_segmented_img,MORPH_CLOSE,getStructuringElement(MORPH_ELLIPSE,Size(3, 3)),
                 Point(-1, -1),1);
    morphologyEx(binary_segmented_img,binary_segmented_img,MORPH_OPEN,getStructuringElement(MORPH_ELLIPSE,Size(3,3)),
                 Point(-1,-1),3);
    cv::namedWindow("Before_Morph");
    cv::imshow("Before_Morph",binary_segmented_img);
    cv::waitKey(0);
    // Hough circles transformation for circle detection on the binary mask
    double min_distance_between_circles = static_cast<double>(binary_segmented_img.cols) / 40;
    int thresh1 = 300;
    int thresh2 = 6;
    double min_radius = static_cast<double>(std::max(binary_segmented_img.cols, binary_segmented_img.rows)) / 115;
    double max_radius = static_cast<double>(std::max(binary_segmented_img.cols, binary_segmented_img.rows)) / 35;
    std::vector<Vec3f> detected_circles;
    HoughCircles(binary_segmented_img,circles,HOUGH_GRADIENT,1,min_distance_between_circles,thresh1,thresh2,
                 min_radius, max_radius);
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



void printCircles(const Mat& img, const std::vector<Vec3f>& circles, int circles_img_size, std::vector<Mat>& circles_img) {
    Mat mask, ball_region;
    for (size_t i = 0; i < circles.size(); ++i) {
        // Define circle center and radius
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);

        // Compute the circle mask and apply it to img
        mask = Mat::zeros(img.size(), CV_8U);
        ball_region = Mat::zeros(img.size(), img.type()); // Ensure ball_region has the same type as img
        circle(mask, center, radius, Scalar(255), -1);
        img.copyTo(ball_region, mask);

        // Convert the image to grayscale
        Mat gray_ball_region;
        cvtColor(ball_region, gray_ball_region, COLOR_BGR2GRAY);

        // Threshold the grayscale image to get a binary mask
        Mat binary_ball_region;
        threshold(gray_ball_region, binary_ball_region, 1, 255, THRESH_BINARY);

        // Find the bounding box of the non-black region (subject)
        Rect boundingBox = boundingRect(binary_ball_region);

        // Crop the image using the bounding box coordinates
        Mat cropped_ball_region = ball_region(boundingBox).clone(); // Ensure we clone the submatrix

        // Create a new black image of the desired fixed size
        Size img_size(circles_img_size, circles_img_size);
        Mat new_circle_img = Mat::zeros(img_size, img.type()); // Ensure new_circle_img has the same type as img

        // Calculate the position to place the subject in the center
        int xOffset = (img_size.width - boundingBox.width) / 2;
        int yOffset = (img_size.height - boundingBox.height) / 2;

        // Ensure valid copy-to operation
        if (xOffset >= 0 && yOffset >= 0 &&
            boundingBox.width <= img_size.width && boundingBox.height <= img_size.height) {
            // Place the cropped subject in the center of the new image
            Mat roi(new_circle_img, Rect(xOffset, yOffset, boundingBox.width, boundingBox.height));
            cropped_ball_region.copyTo(roi);
        }

        // Add the new image to the circles_img vector
        circles_img.push_back(new_circle_img);
    }
}
