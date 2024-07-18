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
    {       // Print an error message using cv::Error
       std::cerr << "Error: " << format("Failed to load image! Error code: %d", cv::Error::StsError) << std::endl;       exit(0);
    }
    //Detection of the billiard balls
	std::vector<Vec3f> circles;
    ballDetection(img,circles);
    std::vector<Mat> circles_images;    int circle_size = 100;
    std::vector<Mat> neighborhoods;    balls_neighbourhood(img,circles,neighborhoods, circles_images);
    //printCircles(img,circles,circle_size,circles_images);    
    // Draw the detected circles    
	Mat circles_img;
    drawCircles(img,circles_img,circles);    std::vector<billiardBall> balls; // vector of object balls
        for(int i = 0; i < circles_images.size(); i++)
    {       balls.emplace_back(circles[i][0],circles[i][1],circles[i][2],10,circles_images[i]);       
    }
    classify(img,neighborhoods, circles_images);    
    imshow("Circles",circles_img);
    waitKey(0);
	destroyAllWindows();
    return balls;
}
/*std::vector<billiardBall> ball_detection(const cv::Mat& inputImage)
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

	balls_neighbourhood(img,circles,circles_images);
//	printCircles(img,circles,circle_size,circles_images);
	
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

*/


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

// Calculate the window size as function of the image size
    int window_size = static_cast<int>(std::round(static_cast<double>(std::max(img.rows,img.cols)) / window_ratio));


	Mat temp = img.clone();
	Mat hsv_img_mean_shift;
    cvtColor(temp, hsv_img_mean_shift, COLOR_BGR2HSV_FULL);
    std::vector<Mat> img_channels_mean_shift;
    split(hsv_img_mean_shift, img_channels_mean_shift);


	Scalar mh_F, sh_F;
    meanStdDev(img_channels_mean_shift[0], mh_F, sh_F);

    Scalar ms_F, ss_F;
    meanStdDev(img_channels_mean_shift[1], ms_F, ss_F);

    Scalar mv_F, sv_F;
    meanStdDev(img_channels_mean_shift[2], mv_F, sv_F);



	double spatial_window_radius = (0.001*window_size); 
	double color_window_radius =  sh_F[0];

	cvtColor(temp,temp,COLOR_BGR2Lab);
	pyrMeanShiftFiltering(temp,temp,spatial_window_radius ,color_window_radius );
	cvtColor(temp,temp,COLOR_Lab2BGR);
	//namedWindow("MeanShift");
	//imshow("MeanShift",temp);
	//waitKey(0);




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
            double lower_coeff = 1.2; // 0.8
            double higher_coeff = 0.9; // 1.2

            double lower_weight[3];
            double higher_weight[3];


			// Hue settings for lower/upper bound differences
			if(most_common_color[0] > mean_hue[0] || mh[0]<120)
			{

				lower_weight[0]= 0.8;
				higher_weight[0] = 1.2;
			} else
			{
				higher_weight[0] = 0.8;
				lower_weight[0] = 1.2;
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
            if(h_cond < 0.3) // case 1: non-uniform window only in Hue
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
            if(s_cond<0.3)
            {
	                hsv_thresholds[2] = static_cast<int>(1000); // Lower bound for saturation
                hsv_thresholds[3] = static_cast<int>(1000); // Upper bound for saturation


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
            if(v_cond<0.4)
            {
			                hsv_thresholds[4] = static_cast<int>(lower_weight[2]*lower_coeff*v_t); // Lower bound for value
                hsv_thresholds[5] = static_cast<int>(higher_weight[2]*lower_coeff*v_t); // Upper bound for value
				


            } else {
					if(mh[0]<120 && h_cond > 0.8 && v_t/60 > 0.8)
					higher_coeff = 1000;
					else higher_coeff = 1;
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
	// closing: filling gaps & connect adjacent objects
    morphologyEx(binary_segmented_img,binary_segmented_img,MORPH_CLOSE,getStructuringElement(MORPH_ELLIPSE,Size(3, 3)),
                 Point(-1, -1),1); // 1
	// opening: brake narrow connection between objects
    morphologyEx(binary_segmented_img,binary_segmented_img,MORPH_OPEN,getStructuringElement(MORPH_ELLIPSE,Size(3,3)),
                 Point(-1,-1),3); // 3
    morphologyEx(binary_segmented_img,binary_segmented_img,MORPH_OPEN,getStructuringElement(MORPH_ELLIPSE,Size(7,7)),
                 Point(-1,-1),1); // 3

    //cv::namedWindow("Before_Morph");
    //cv::imshow("Before_Morph",binary_segmented_img);
    //cv::waitKey(0);
    // Hough circles transformation for circle detection on the binary mask
    double min_distance_between_circles = static_cast<double>(binary_segmented_img.cols) / 40; // 40
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

bool isCircular(vector<Point> contour) {
    double area = contourArea(contour);
    double perimeter = arcLength(contour, true);
    double circularity = 4 * CV_PI * (area / (perimeter * perimeter));
	return circularity > 0.4;  // threshold for circularity
}

void classify(const Mat& img, std::vector<Mat>& neighborhoods, std::vector<Mat>& circles_img) {

    // GLOBAL EVALUATION

    // Convert the image to HSV color space
    Mat hsv_img;
    cvtColor(img, hsv_img, COLOR_BGR2HSV_FULL);
    std::vector<Mat> img_channels;
    split(hsv_img, img_channels);

    // Compute the mean and standard deviation for each channel
    Scalar mh, sh;
    meanStdDev(img_channels[0], mh, sh);

    Scalar ms, ss;
    meanStdDev(img_channels[1], ms, ss);

    Scalar mv, sv;
    meanStdDev(img_channels[2], mv, sv);

    //DISCARD BASED ON MOST COMMON COLOR
    Vec3b most_common_color;
    mostCommonColor(hsv_img,most_common_color);

    // IMAGE BALL EVALUATION

    for(size_t i = 0; i < neighborhoods.size(); ++i){



        // Mean and Standard deviation of the Neighborhood

        Mat hsv_window;
        cvtColor(neighborhoods[i], hsv_window, COLOR_BGR2HSV_FULL);
        split(hsv_window, img_channels);

        // Compute the mean and standard deviation for each channel
        Scalar wmh, wsh;
        meanStdDev(img_channels[0], wmh, wsh);

        Scalar wms, wss;
        meanStdDev(img_channels[1], wms, wss);

        Scalar wmv, wsv;
        meanStdDev(img_channels[2], wmv, wsv);
        Vec3b w_most_common_color;
        mostCommonColor(hsv_window,w_most_common_color);

        // Mean and Standard deviation of the Ball

        Mat hsv_ball;
        cvtColor(circles_img[i], hsv_ball, COLOR_BGR2HSV_FULL);
        split(hsv_ball, img_channels);


        // Compute the mean and standard deviation for each channel
        Scalar bmh, bsh;
        meanStdDev(img_channels[0], bmh, bsh);

        Scalar bms, bss;
        meanStdDev(img_channels[1], bms, bss);

        Scalar bmv, bsv;
        meanStdDev(img_channels[2], bmv, bsv);
        Vec3b b_most_common_color;
        mostCommonColor(hsv_ball,b_most_common_color);

		// FAZZI's method ------------------

		Mat image = circles_img[i];
		Mat gray, blurred, edged;

		//cvtColor(image,image,COLOR_BGR2Lab);
		//pyrMeanShiftFiltering(image,image,1,1);
		//cvtColor(image,image,COLOR_Lab2BGR);

		cvtColor(image, gray, COLOR_BGR2GRAY);
		GaussianBlur(gray, blurred, Size(3, 3),bsh[0],bsh[0]);
		Canny(blurred, edged, 50, 150);
		Size newSize(100, 100);  // Width and height

		// Resize the image
		Mat resizedCanny;
		resize(edged, resizedCanny, newSize);


		vector<vector<Point>> contours;
		findContours(edged, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		for (auto &contour : contours) {
			if (isCircular(contour)) {
				// Accept the contour as a ball
				//drawContours(image, vector<vector<Point>>{contour}, -1, Scalar(0, 255, 0), 0.2);
			} else {
				// Reject the contour as a false positive
				//drawContours(image, vector<vector<Point>>{contour}, -1, Scalar(0, 0, 255), 0.2);
			}
		}
        //CLASSIFICATION

        int a = abs(sh[0] - wsh[0]);
        int b = abs(ss[0] - wss[0]);
        int c = abs(sv[0] - wsv[0]);
        double STDdistance = sqrt(a * a + b * b + c * c);
        cout << STDdistance << endl;

        int discard_count = 0;

        if(STDdistance<70) {
			// Specify the new size
			Size newSize(100, 100);  // Width and height

			// Resize the image
			Mat resizedImage;
			resize(circles_img[i], resizedImage, newSize);
	imshow("cannyImg",resizedCanny);
	waitKey(0);

            imshow("True_Positives",resizedImage);
            waitKey(0);
            //Mat save = circles_img[i].clone(); // Clone the region to store in the vector
            //
            //circles_img.push_back(save); // Store the region of interest in the vector
        }

        else{
            //discard_count++;
            //imshow("FALSE POSITIVE",circles_img[i]);
            //waitKey(0);
        }
        //cout << "Number of discarded balls: " << endl;
        //cout << discard_count << endl;
    }
}

void balls_neighbourhood(const Mat& img, const std::vector<Vec3f>& circles, std::vector<Mat>& neighborhoods, std::vector<Mat>& circles_img) {
    double x, y;
    double radius;
    Rect window;
    Rect ball;
    namedWindow("Circles");
    for(int i = 0; i < circles.size(); i++)
    {
        x = cvRound(circles[i][0]);
        y = cvRound(circles[i][1]);
        radius = cvRound(circles[i][2]);
        int window_dim = 4;

        // Define the window size
        window.height = radius * 2 *window_dim;
        window.width = radius * 2 *window_dim;

        // Calculate the top-left corner of the window
        window.x = x - window.width/2;
        window.y = y - window.height/2;

        // Adjust window dimensions and position to ensure it stays within image bounds
        if (window.x < 0) {
            window.width += window.x;
            window.x = 0;
        }
        if (window.y < 0) {
            window.height += window.y;
            window.y = 0;
        }
        if (window.x + window.width > img.cols) {
            window.width = img.cols - window.x;
        }
        if (window.y + window.height > img.rows) {
            window.height = img.rows - window.y;
        }
        Mat neighborhood = img(window).clone(); // Clone the region to store in the vector
        Point center(cvRound(window.width/2), cvRound(window.height/2));
        circle(neighborhood,center,radius+1,Scalar(0, 0, 0),-1, LINE_AA);
        neighborhoods.push_back(neighborhood); // Store the region of interest in the vector
        //imshow("FALSE POSITIVE",neighborhood);
        //waitKey(0);
		double ball_dim = 1.7;

        // Define the window size
        ball.height = radius * 2 *ball_dim;
        ball.width = radius * 2 *ball_dim;


        // Calculate the top-left corner of the window
        ball.x = x - ball.width/2;
        ball.y = y - ball.height/2;

        // Adjust window dimensions and position to ensure it stays within image bounds
        if (ball.x < 0) {
            ball.width += ball.x;
            ball.x = 0;
        }
        if (ball.y < 0) {
            ball.height += ball.y;
            ball.y = 0;
        }
        if (ball.x +ball.width > img.cols) {
            ball.width = img.cols - ball.x;
        }
        if (ball.y + ball.height > img.rows) {
            ball.height = img.rows - ball.y;
        }
        circles_img.push_back(img(ball)); // Store the region of interest in the vector
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
