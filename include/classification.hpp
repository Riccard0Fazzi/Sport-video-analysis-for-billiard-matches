#ifndef CLASSIFICATION
#define CLASSIFICATION

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <iostream>

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

cv::Mat extractFeatures(const Mat& image) {

int predict(const Ptr<ml::SVM>& svm, const Mat& image) {

#endif // CLASSIFICATION
