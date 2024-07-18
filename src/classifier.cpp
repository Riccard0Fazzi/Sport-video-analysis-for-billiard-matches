#include "field_detection.h"
#include "ball_detection.hpp"

using namespace cv;
using namespace std;

void classify(vector<billiardBall> &balls, Mat& table_image)
{
	vector<billiardBall> copy_of_balls = balls;
	Vec3b table_most_common_color; // this is the most common color in BGR of the table
	Vec3b ball_most_common_color; // this is the most common color in BGR of the bal
	Mat field_clone = table_image.clone();
	cvtColor(field_clone,field_clone,COLOR_BGR2HSV_FULL);
	Mat temp_ball;

    mostCommonColor(field_clone,table_most_common_color); 
    Mat rgb_mat(100, 100, CV_8UC3, table_most_common_color);  
	cvtColor(rgb_mat,rgb_mat,COLOR_HSV2BGR_FULL);
 
     Mat mask_black;
	
	Vec3b black = {0,0,0};
	cout << "Table_most_common_color: " << table_most_common_color << endl; 
	namedWindow("Balls_Filtered");
	namedWindow("Balls_Original");
	// iterate through each ball and try to classify it
	for(int i = 0; i < copy_of_balls.size(); i++)
	{

		temp_ball = copy_of_balls[i].image;
	//	mostCommonColor(temp_ball,ball_most_common_color);
		inRange(temp_ball,Scalar(0,0,0),Scalar(table_most_common_color[0],table_most_common_color[1],table_most_common_color[2]),mask_black);
		temp_ball.setTo(black,mask_black);
		
		imshow("Balls_Filtered",temp_ball);
		imshow("Balls_Original",copy_of_balls[i].image);

		waitKey(0);
	}

}
