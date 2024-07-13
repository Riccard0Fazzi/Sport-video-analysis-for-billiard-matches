#ifndef CLASSIFICATION
#define CLASSIFICATION

#include <iostream>
#include <opencv2/opencv.hpp>


struct BilliardBall {
    int x;
    int y;
    int width;
    int height;
    int ID;
    cv::Mat image;

    // Optional: You can add a constructor for initialization
    BilliardBall(int x, int y, int width, int height, int ID, const cv::Mat& img)
        : x(x), y(y), width(width), height(height), ID(ID), image(img.clone()) {}
};

// Example usage
BilliardBall ball(10, 20, 30, 30, 1, cv::imread("ball_image.jpg"));


#endif // CLASSIFICATION
