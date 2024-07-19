
bool isCircular(vector<Point> contour, Size size) {
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

    // ALMOST CLOSED CIRCLES CONDITION
    if(isContourConvex(approx)==false){

        // IF IT'S IN THE CENTER
        if((abs(center.x- size.width/2)<3)&&(abs(center.y- size.height/2)<5)){
            //cout << "OUT OF CENTER" << endl;
            //cout << abs(center.x- size.width/2) << " " << abs(center.y- size.height/2) << endl;
            return true;
        }
        //cout << "DEBUG" << endl;
        //cout << abs(center.x- size.width/2) << " " << abs(center.y- size.height/2) << endl;

        return false;
    }
    // CLOSED CIRCLES CONDITION
    if((abs(center.x- size.width/2)>3)&&(abs(center.y- size.height/2)>5)){

        //cout << abs(center.x- size.width/2) << " " << abs(center.y- size.height/2) << endl;
        return false;
    }



    return true;
    /*

    // ALMOST A CIRCLE
    if(isContourConvex(approx)==false){
        if(radius > 2.5 && radius < 13) {
            if((abs(center.x- size.width/2)<3)&&(abs(center.y- size.height/2)<5))
            cout << abs(center.x- size.width/2) << " " << abs(center.y- size.height/2) << endl;
            return true;
        }
    }

    // CIRCLE
    if(radius > 2.5 && radius < 13) {
        cout << abs(center.x- size.width/2) << " " << abs(center.y- size.height/2) << endl;
        return true;
    }
    return false;*/

    /*
    if(radius > 2.5 && radius < 13){
        // is it not too small ?
        if (contour.size() < 8 || contour.size() > 20) {
            return false;
        }
        if((abs(center.x- size.width/2)>3)&&(abs(center.y- size.height/2)>5)){
            //cout << "OUT OF CENTER" << endl;
            //cout << abs(center.x- size.width/2) << " " << abs(center.y- size.height/2) << endl;
            return false;
        }
        return true;
    }
    return false;*/


    /*
    // is it convex ?

    if(isContourConvex(approx)){

        // is it not too small ?
        if (contour.size() < 8) {
            return false;
        }

        // is it in the middle of the window ????
        double x_sum = 0.0;
        double y_sum = 0.0;
        std::vector<Point> convexHull;
        cv::convexHull(contour, convexHull);

        for (const auto& point : convexHull) {
            x_sum += point.x;
            y_sum += point.y;
        }

        double num_points = static_cast<double>(convexHull.size());
        Point2d centroid = { x_sum / num_points, y_sum / num_points };
        if((abs(centroid.x- size.width/2)>3)&&(abs(centroid.y- size.height/2)>5)){
            //cout << "OUT OF CENTER" << endl;
            return false;
        }

        return true;
    }
    return false;*/

    /*
    if(distance<3){
        if(area>4.1){
            if(circularity<0.02){
                return false;
            }
            cout << area << endl;
            return true;
        }
        else{

            return false;
        }
    }
	return false; //circularity > nsh*nsv;  // threshold for circularity
     */
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
        double nsh;
        if(bsh[0]>sh[0]){
            nsh = sh[0]/bsh[0];
        }
        else{
            nsh = bsh[0]/sh[0];
        }
        double nsv;
        if(bsv[0]>sv[0]){
            nsv = sv[0]/bsv[0];
        }
        else{
            nsv = bsv[0]/sv[0];
        }
        Mat otsu_thresh_image;
        Mat discard;

        // BLACK BALLS TRACTATION

        bool color_cond = abs(b_most_common_color[0] - most_common_color[0])<50;
        cout << "COLOR COND " << color_cond << endl;
        bool center_black_cond = img_channels[2].at<uchar>(img_channels[2].rows/2,img_channels[2].cols/2) < 65;
        cout << "CENTER BLACK COND " << static_cast<int>(img_channels[2].at<uchar>(img_channels[2].rows/2,img_channels[2].cols/2)) << endl;
        double otsu_thresh_h;
        if(color_cond && center_black_cond){
            otsu_thresh_h = cv::threshold(img_channels[2], otsu_thresh_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
            morphologyEx(otsu_thresh_image,otsu_thresh_image,MORPH_OPEN,getStructuringElement(MORPH_ELLIPSE,Size(3,3)),
                         Point(-1,-1),1); // 3
            cout << "DEBUG" << endl;
        }
        else{
            otsu_thresh_h = cv::threshold(img_channels[0], otsu_thresh_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        }

        morphologyEx(otsu_thresh_image,otsu_thresh_image,MORPH_CLOSE,getStructuringElement(MORPH_ELLIPSE,Size(3, 3)),
                     Point(-1, -1),1); // 1
        // opening: brake narrow connection between objects
        morphologyEx(otsu_thresh_image,otsu_thresh_image,MORPH_OPEN,getStructuringElement(MORPH_ELLIPSE,Size(3,3)),
                     Point(-1,-1),3); // 3
        double TL = otsu_thresh_h*nsv;
        double TH = 2*TL;
        Canny(otsu_thresh_image, edged, TL, TH); // blurred
        Size newSize(100, 100);  // Width and height

        // Resize the image
        Mat resizedCanny;
        resize(otsu_thresh_image, resizedCanny, newSize); // edged


        vector<vector<Point>> contours;
        findContours(edged, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        cout << "---------" << endl;
        //cout << contours.size() << endl;
        int a = abs(sh[0] - wsh[0]);
        int b = abs(ss[0] - wss[0]);
        int c = abs(sv[0] - wsv[0]);
        double STDdistance = sqrt(a * a + b * b + c * c);
        for (auto &contour : contours) {
            if (isCircular(contour, img_channels[0].size())) {
                cv::Point2f center;
                float radius;
                cv::minEnclosingCircle(contour, center, radius);
                circle(image, center, radius, Scalar(203, 192, 255), 1, LINE_AA);
                // Accept the contour as a ball
                drawContours(image, vector<vector<Point>>{contour}, -1, Scalar(0, 255, 0), 0.2);
            } else {
                // Reject the contour as a false positive
                drawContours(image, vector<vector<Point>>{contour}, -1, Scalar(0, 0, 255), 0.2);
            }
            Size newSize(100, 100);  // Width and height


        }
        // Resize the image
        Mat resizedImage;
        resize(circles_img[i], resizedImage, newSize);
        //imshow("cannyImg", resizedCanny);

        //imshow("True_Positives", resizedImage);
        //waitKey(0);
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
        double ball_dim = 2.5;

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
