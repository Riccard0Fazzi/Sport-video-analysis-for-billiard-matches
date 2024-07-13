#ifndef TRAINING_MODEL
#define TRAINING_MODEL

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <iostream>

cv::Mat extractFeatures(const Mat& image);
void loadData(std::vector<Mat>& images, std::vector<int>& labels);
Ptr<ml::SVM> trainSVM(const std::vector<Mat>& trainData, const std::vector<int>& trainLabels);
int predict(const Ptr<ml::SVM>& svm, const Mat& image);

#endif // CLASSIFICATION
