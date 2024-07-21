#include "ball_detection.hpp"
using namespace cv;
using namespace std;

// Structure used to store balls names and colors
billiardBall::billiardBall(int x, int y, double true_radius, int id, cv::Mat& ballImage)
    : x(x), y(y), true_radius(true_radius),id(id), ballImage(ballImage)
{
    // Optionally, you can add additional initialization logic here if needed
}

void billiardBall::createBoundingBox(cv::Point offset) {
        int x_corner = this->x - static_cast<int>(this->true_radius)*1.5 + offset.x;
        int y_corner = this->y - static_cast<int>(this->true_radius)*1.5 + offset.y;
        box = cv::Rect (x_corner, y_corner, 3 * static_cast<int>(this->true_radius), 3 * static_cast<int>(this->true_radius));
    }



std::vector<billiardBall> ball_detection(const cv::Mat& inputImage, Point offset)
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
    std::vector<Mat> circles_images;
    std::vector<cv::Point2f> centers_window;
    balls_neighbourhood(img,circles, circles_images, centers_window);
    // Draw the detected circles on the image
	Mat circles_on_field_to_print;
    drawCircles(img,circles_on_field_to_print,circles);
 	std::vector<billiardBall> balls; // vector of object balls
    for(int i =0;i<circles.size();i++){
        balls.emplace_back(circles[i][0],circles[i][1],circles[i][2],-1,circles_images[i]); // -1 NOT ASSIGNED YET
		balls[i].createBoundingBox(offset);
    }
    // REMOVE FALSE POSITIVES
    discardFalsePositives(img,centers_window,balls);

    // VISUALIZE DETECTED BALLS
    Mat detected_balls = img.clone();
    for(int i =0;i<balls.size();i++){
        Point center (balls[i].x ,balls[i].y);
        circle(detected_balls,center,balls[i].true_radius,Scalar(0, 255, 0),2, LINE_AA);
    }

    imshow("DETECTED BALLS",detected_balls);
    waitKey(0);
	//destroyAllWindows();
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
    //Mat mask;
    //inRange(img, Scalar(0, 0, 0), Scalar(0, 0, 1), mask); // Only mask black pixels
    //bitwise_not(mask, mask); // Invert mask to include non-black pixels

    // Compute the histogram
    Mat hist;
    //calcHist(&img, 1, channels, mask, hist, 3, hist_size, ranges);
    calcHist(&img, 1, channels, Mat(), hist, 3, hist_size, ranges);
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

    // Initialize the destination image
    dest = Mat::zeros(img.size(), img.type());

    // Determine the most common color in the window region by histogram evaluation
    Vec3b field_color;
    mostCommonColor(img,field_color);
    window_ratio = window_ratio + sh[0];

// Calculate the window size as function of the image size
    int window_size = static_cast<int>(std::round(static_cast<double>(std::max(img.rows,img.cols)) / window_ratio));



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






    // COLOR VARIATION

    // Apply Sobel operator in Y direction
    cv::Mat grad_x;
    cv::Sobel(img_channels[0], grad_x, CV_64F, 1, 0, 3);

    // Apply Sobel operator in Y direction
    cv::Mat grad_y;
    cv::Sobel(img_channels[0], grad_y, CV_64F, 0, 1, 3);

    // Compute the gradient magnitude
    cv::Mat hue_mag;
    cv::magnitude(grad_x, grad_y, hue_mag);

    // BRIGHTNESS VARIATION

    // Apply Sobel operator in Y direction
    cv::Sobel(img_channels[2], grad_x, CV_64F, 1, 0, 3);

    // Apply Sobel operator in Y direction
    cv::Sobel(img_channels[2], grad_y, CV_64F, 0, 1, 3);

    // Compute the gradient magnitude
    cv::Mat val_mag;
    cv::magnitude(grad_x, grad_y, val_mag);


    // Iterate over the image with non-overlapping windows
    for (int y = 0; y < img.rows; y += window_size) {
        for (int x = 0; x < img.cols; x += window_size) {

            // Define the window region, ensuring it doesn't exceed the image bounds
            int window_width = std::min(window_size, img.cols - x);
            int window_height = std::min(window_size, img.rows - y);
            Rect window(x, y, window_width, window_height);
            Mat window_region = hsv_img(window);
            Mat window_hue_mag = hue_mag(window);
            Mat window_val_mag = val_mag(window);

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


            // Create a mask for the most common color in the window region

            //  MEAN AND STD OF COLOR VARIATION IN SUBWINDOW
            Scalar m_hue_mag, s_hue_mag;
            meanStdDev(window_hue_mag, m_hue_mag, s_hue_mag);

            //  MEAN AND STD OF BRIGHTNESS VARIATION IN SUBWINDOW
            Scalar m_val_mag, s_val_mag;
            meanStdDev(window_val_mag, m_val_mag, s_val_mag);



            int HUE_THRESH = 10; //10;
            int SAT_THRESH = 55; //60;
            int VAL_THRESH = 66; //+ 0.1*s_val_mag[0]; //60;



            Scalar lower_bound(
                    std::max(most_common_color[0] - HUE_THRESH, 0),
                    std::max(most_common_color[1] - SAT_THRESH, 0),
                    std::max(most_common_color[2] - VAL_THRESH, 0)
            );
            Scalar upper_bound(
                    std::min(most_common_color[0] + HUE_THRESH, 180),
                    std::min(most_common_color[1] + SAT_THRESH, 255),
                    std::min(most_common_color[2] + VAL_THRESH, 255)
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
    double window_ratio = 10; //14.6; //40
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

    morphologyEx(binary_segmented_img,binary_segmented_img,MORPH_CLOSE,getStructuringElement(MORPH_ELLIPSE,Size(3, 3)),Point(-1, -1),1); // 1
																																		 
    morphologyEx(binary_segmented_img,binary_segmented_img,MORPH_OPEN,getStructuringElement(MORPH_ELLIPSE,Size(3,3)),Point(-1,-1),4); // 3
	// opening: brake narrow connection between objects
   // morphologyEx(binary_segmented_img,binary_segmented_img,MORPH_OPEN,getStructuringElement(MORPH_ELLIPSE,Size(3,3)),Point(-1,-1),1); // 3


    //cv::namedWindow("Before_Morph");
    //cv::imshow("Before_Morph",binary_segmented_img);
    //cv::waitKey(0);
    // Hough circles transformation for circle detection on the binary mask
    double min_distance_between_circles = static_cast<double>(binary_segmented_img.cols) / 35; // 40
    int thresh1 = 300;
    int thresh2 = 6;//6
    double min_radius = static_cast<double>(std::max(binary_segmented_img.cols, binary_segmented_img.rows)) / 105;//115
    double max_radius = static_cast<double>(std::max(binary_segmented_img.cols, binary_segmented_img.rows)) / 55;//35    55
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

// Function to calculate the Euclidean distance between two points
double distance(const Point& p1, const Point& p2) {
    return (abs(p1.x - p2.x)  + abs(p1.y - p2.y));
}

bool isBall(vector<Point> contour, Point2f c, Mat hsv_image) {
    double area = contourArea(contour);
    double perimeter = arcLength(contour, true);
    double circularity = 4 * CV_PI * (area / (perimeter * perimeter));
    // Convexity conditions

    std::vector<Point> approx;
    double arc_len = cv::arcLength(contour, true);
    double almost_closed = 0.1;
    cv::approxPolyDP(contour, approx, almost_closed * arc_len, true);

    // almost closed circles
    cv::Point2f center;
    float radius;
    cv::minEnclosingCircle(contour, center, radius);


    // Initialize the minimum distance with a large value
    double min_distance = DBL_MAX;

    // Loop through all points in the contour
    for (const Point& point : contour) {
        // Calculate the distance from the contour point to the center point
        double d = distance(point, c);

        // Update the minimum distance if a smaller distance is found
        if (d < min_distance) {
            min_distance = d;
        }
    }
    // EVALUATE CENTER OF IMAGE
    // Define the rectangle
    cv::Rect rect(c.x, c.y, 5, 5);

    // Crop the image using the rectangle
    cv::Mat centerImage = hsv_image(rect);
    Vec3b most_common_color;
    mostCommonColor(hsv_image,most_common_color);
    Vec3b most_common_color_center;
    mostCommonColor(centerImage,most_common_color_center);
    std::vector<Mat> img_channels;
    split(hsv_image, img_channels);
    Mat value = img_channels[2];
    double distance = cv::norm(most_common_color - most_common_color_center);

    // Compute the mean and standard deviation for each channel
    Scalar mh, sh;
    meanStdDev(img_channels[0], mh, sh);

    Scalar ms, ss;
    meanStdDev(img_channels[1], ms, ss);

    Scalar mv, sv;
    meanStdDev(img_channels[2], mv, sv);

    double s = sqrt(sh[0] *sh[0] + ss[0] * ss[0] + sv[0] * sv[0]);

    // IS IT INSIDE THE CONTOUR

    if(pointPolygonTest(contour, c, false)>0){


        // DISCARD FALSE POSITIVES LOOKING FOR REAL CIRCLES


        // discard fingers

        if(s>80 && abs(most_common_color[0]-most_common_color_center[0])<100) {
            return false;
        }
        // KEEP BLACK BALLS far value but close sat
        if((abs(most_common_color[2]-most_common_color_center[2])>160)&&(abs(most_common_color[1]-most_common_color_center[1])<60)){
            return true;
        }
        // DISCARD HOLES far value but close hue
        if(abs(most_common_color[2]-most_common_color_center[2])>160 && abs(most_common_color[0]-most_common_color_center[0])<60){

            return false;
        }

        // discard ground points
        // Calculate the Euclidean norm (distance) between the two points
        if(distance<8 && s < 50&& circularity<0.8){
            return false;
        }

        return true;
    }

    // IS IT CIRCULAR ENOUGH AND MIN DISTANCE IS SMALL ??
    else if(min_distance<=4 && circularity>0.2) { // 3 , 0.3
        return true;
    }

    /*
    else if(min_distance<=10) {
        if(abs(most_common_color[1]-most_common_color_center[1])>150){
            cout << radius << endl;
            Mat show;
            cvtColor(hsv_image, show, COLOR_HSV2BGR_FULL);
            imshow(" ",show);
            waitKey();
        }
    }*/

    else{


        /*
        cout << abs(most_common_color[1]-most_common_color_center[1]) << endl;
        Mat show;
        cvtColor(hsv_image, show, COLOR_HSV2BGR_FULL);
        imshow(" ",show);
        waitKey();*/
        return false;
    }







    /*
    std::vector<Mat> img_channels;
    split(hsv_image, img_channels);
    int hole_cond = img_channels[2].at<uchar>(center);

    if(isContourConvex(approx)&& hole_cond >150){
        return true;
    }*/

    // if convex and not far as

    /*

    // TOO BIG CONDITION
    // area is not a good index, really big contours have still small areas somehow

    // IS IT CENTERED ?
    if(abs(center.x- c.x)<5&&abs(center.y- c.y)<5) {
        cout << "AREA " << endl;
        cout << area << endl;
        return true;
    }
    // IF NOT CENTERED BUT CLOSEST POINT TO THE CENTER IS CLOSE AND CONVEX CONTOUR
    if(min_distance<=3 && isContourConvex(approx)){
        return true;
    }

    // NOT UPPER PROPERTIES BUT CENTER IS INSIDE CONTOUR (SET AS INITIAL CONDITION ?)
    if(pointPolygonTest(contour, c, false)>0){
        return true;
    }
    cout << "MIN DISTANCE " << endl;
    cout << min_distance << endl;
    cout << "NOT CLOSE" << endl;*/
    return false;
}

void discardFalsePositives(const Mat& img,std::vector<cv::Point2f>& centers,std::vector<billiardBall>& balls) {

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

    for(size_t i = 0; i < balls.size(); i++){


        // Mean and Standard deviation of the Ball

        Mat hsv_ball;
        cvtColor(balls[i].ballImage, hsv_ball, COLOR_BGR2HSV_FULL);
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

        Mat image = balls[i].ballImage.clone();
        //Mat gray, blurred, edged;
        Mat gray, edged;
        //cvtColor(image,image,COLOR_BGR2Lab);
        //pyrMeanShiftFiltering(image,image,1,1);
        //cvtColor(image,image,COLOR_Lab2BGR);

        //cvtColor(img_channels[0], gray, COLOR_BGR2GRAY);
        //GaussianBlur(img_channels[0], blurred, Size(3, 3),0,0);
        Mat hue = img_channels[0];

        Mat thresh_image;
        double otsu_thresh;


        // THRESHOLDING VARIANTS
        // ---------------------------------------------------------

        /*
        // OTSU's on GRAYSCALE
        Mat blurred;
        Mat clustered;
        cvtColor(image,clustered,COLOR_BGR2Lab);
        int adaptive_kernel = static_cast<int>(min(image.rows,image.cols)/5);
        if (adaptive_kernel%2 == 0){
            adaptive_kernel++;
        }
        pyrMeanShiftFiltering(clustered,clustered,adaptive_kernel,adaptive_kernel);
        cvtColor(clustered,clustered,COLOR_Lab2BGR);
        cvtColor(clustered, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, blurred, Size(adaptive_kernel, adaptive_kernel),bsh[0],bsh[0]);
        otsu_thresh = cv::threshold(blurred, thresh_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        */


        // OTSU's on H
        int adaptive_kernel = static_cast<int>(min(image.rows,image.cols)/5);
        if (adaptive_kernel%2 == 0){
            adaptive_kernel++;
        }
        GaussianBlur(hue, hue, Size(adaptive_kernel, adaptive_kernel),bsh[0],bsh[0]);
        otsu_thresh = cv::threshold(hue, thresh_image, 0, 180, cv::THRESH_BINARY | cv::THRESH_OTSU);



        // ADAPTIVE MEAN

        // Positive values of C will make the threshold value lower, meaning more pixels will be considered white
        // and negative values will make it higher, meaning fewer pixels will be considered foreground.
        /*
        int adaptive_kernel = static_cast<int>(min(image.rows,image.cols)/5);
        if (adaptive_kernel%2 == 0){
            adaptive_kernel++;
        }
        double C = 5;
        adaptiveThreshold(hue, thresh_image, 180, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, adaptive_kernel, C);
         */


        morphologyEx(thresh_image,thresh_image,MORPH_CLOSE,getStructuringElement(MORPH_ELLIPSE,Size(3, 3)),
                     Point(-1, -1),1); // 1
        // opening: brake narrow connection between objects
        morphologyEx(thresh_image,thresh_image,MORPH_OPEN,getStructuringElement(MORPH_ELLIPSE,Size(3,3)),
                     Point(-1,-1),3); // 3
        double TL = otsu_thresh;
        double TH = 2*TL;

        Canny(thresh_image, edged, TL, TH); // blurred
        Size newSize(400, 400);  // Width and height

        // Resize the image
        Mat resizedMask;
        resize(thresh_image, resizedMask, newSize); // edged


        vector<vector<Point>> contours;
        findContours(edged, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        // Draw a marker at the center of the image for visualization
        drawMarker(image, centers[i], Scalar(255, 255, 255), MARKER_CROSS, 7, 1);
        bool one_green_detected = false;
        bool one_red_detected = false;

        for (auto &contour : contours) {

            if (isBall(contour, centers[i], hsv_ball)) {

                // one green was detected
                one_green_detected = true;
                cv::Point2f center;
                float radius;
                cv::minEnclosingCircle(contour, center, radius);
                circle(image, center, radius, Scalar(255, 0, 0), 2, LINE_AA);
                circle(image, center, 1.5, Scalar(0, 255, 255), -1, LINE_AA);

                // Accept the contour as a ball
                drawContours(image, vector<vector<Point>>{contour}, -1, Scalar(0, 255, 0), 0.2);

                // UPDATE OF BOUNDING BOX

            } else {
                // one red detected condition
                one_red_detected = true;
                // Reject the contour as a false positive
                drawContours(image, vector<vector<Point>>{contour}, -1, Scalar(0, 0, 255), 0.2);
            }

        }


        // OTHER METHODS

        /*
        // THRESHOLDING - OTSU's on THREE CHANNELS

        vector<Mat> otsu_thresh_channels(3);
        vector<double> otsu_thresh(3);
        vector<Mat> edgeds(3);
        vector<vector<Point>> contours;
        double TL;
        double TH;
        bool one_green_detected = false;
        bool one_red_detected = false;

        for(size_t j = 0; j<3;j++){
            otsu_thresh[j] = cv::threshold(img_channels[j], otsu_thresh_channels[j], 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
            morphologyEx(otsu_thresh_channels[j],otsu_thresh_channels[j],MORPH_CLOSE,getStructuringElement(MORPH_ELLIPSE,Size(3, 3)),
                         Point(-1, -1),1); // 1
            // opening: brake narrow connection between objects
            morphologyEx(otsu_thresh_channels[j],otsu_thresh_channels[j],MORPH_OPEN,getStructuringElement(MORPH_ELLIPSE,Size(3,3)),
                         Point(-1,-1),3);
            TL = otsu_thresh[j];
            TH = 2*TL;
            Canny(otsu_thresh_channels[j], edgeds[j], TL, TH); // blurred
            findContours(edgeds[j], contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            for (auto &contour : contours) {

                if (isBall(contour, centers[i])) {

                    // one green was detected
                    one_green_detected = true;

                    // Accept the contour as a ball
                    drawContours(image, vector<vector<Point>>{contour}, -1, Scalar(0, 255, 0), 0.2);

                    // UPDATE OF BOUNDING BOX

                } else {
                    // one red detected condition
                    one_red_detected = true;
                    // Reject the contour as a false positive
                    drawContours(image, vector<vector<Point>>{contour}, -1, Scalar(0, 0, 255), 0.2);
                }

            }
            contours.clear();

        }
        otsu_thresh_channels.clear();
        otsu_thresh.clear();
        edgeds.clear();
         */
        /*




         // BLACK BALLS TRACTATION

        bool color_cond = abs(b_most_common_color[0] - most_common_color[0])<50;
        //cout << "COLOR COND " << color_cond << endl;
        bool center_black_cond = img_channels[2].at<uchar>(img_channels[2].rows/2,img_channels[2].cols/2) < 65;
        //cout << "CENTER BLACK COND " << static_cast<int>(img_channels[2].at<uchar>(img_channels[2].rows/2,img_channels[2].cols/2)) << endl;

        double otsu_thresh_h;
        if(color_cond && center_black_cond){
            otsu_thresh_h = cv::threshold(img_channels[2], otsu_thresh_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
            morphologyEx(otsu_thresh_image,otsu_thresh_image,MORPH_OPEN,getStructuringElement(MORPH_ELLIPSE,Size(3,3)),
                         Point(-1,-1),1); // 3
            //cout << "DEBUG" << endl;
        }
        else{
            otsu_thresh_h = cv::threshold(img_channels[0], otsu_thresh_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        }
        otsu_thresh_h = cv::threshold(img_channels[0], otsu_thresh_image, 0, 180, cv::THRESH_BINARY | cv::THRESH_OTSU);
        */





        // REMOVAL CONDITION

        // IF NO GREEN DETECTED AND AT LEAST ONE RED DETECTED OR NOTHING DETECTED
        if((one_green_detected == false && one_red_detected == true) || (one_green_detected == false && one_red_detected == false) ){
            // REMOVE FROM BALLS
            balls.erase(balls.begin() + i);
            centers.erase(centers.begin() + i);
            // MAINTAIN TOTAL ORDER IN BALLS
            i--;
        }
        //Size newSize(400, 400);

        // Resize the image
        Mat resizedImage;
        resize(image, resizedImage, newSize);
        //imshow("MASK", resizedMask);

        //imshow("GREEN -> True_Positives", resizedImage);
        //waitKey(0);

    }

}

void balls_neighbourhood(const Mat& src, const std::vector<Vec3f>& circles, std::vector<Mat>& circles_images,std::vector<cv::Point2f>& centers) {
    double x, y;
    double radius;
    Rect ball;
    Mat img = src.clone();
    // Define the size of the border to be added (e.g., 10 pixels on each side)
    Vec3b most_common_color;
    mostCommonColor(img,most_common_color);
    int borderSize = 100;

    // Create a new image with added borders
    copyMakeBorder(img, img, borderSize, borderSize, borderSize, borderSize, BORDER_REPLICATE);

    // COMPUTE MAX RADIUS FOR THAT IMAGE
    double max_radius = 0;
    for(int i = 0; i < circles.size(); i++){
        if (circles[i][2] > max_radius){
            max_radius = circles[i][2];
        }
    }

    for(int i = 0; i < circles.size(); i++)
    {
        x = cvRound(circles[i][0])+borderSize;
        y = cvRound(circles[i][1])+borderSize;
        radius = cvRound(max_radius); //cvRound(circles[i][2]);

        //imshow("FALSE POSITIVE",neighborhood);
        //waitKey(0);
        double ball_dim = 2;

        // Define the window size
        ball.height = radius * 2 *ball_dim;
        ball.width = radius * 2 *ball_dim;


        // Calculate the top-left corner of the window
        ball.x = x - ball.width/2;
        ball.y = y - ball.height/2;

        Mat image = img(ball).clone();

        circles_images.push_back(image); // Store the region of interest in the vector
        Point2f c (ball.width/2,ball.height/2);
        centers.push_back(c);
        // Draw a marker at the center of the image for visualization
        //drawMarker(image, c, Scalar(255, 255, 255), MARKER_CROSS, 7, 1);
        //imshow("Yo",image);
        //waitKey();
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
