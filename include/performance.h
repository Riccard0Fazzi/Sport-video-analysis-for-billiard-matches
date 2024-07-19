// Created by Tommaso Tubaldo on 18/07/24.

#ifndef BILIARDVIDEOANALYSIS_PERFORMANCE_H
#define BILIARDVIDEOANALYSIS_PERFORMANCE_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <fstream>
#include <sstream>

struct BoundingBox {
    int class_id;
    cv::Rect box;
};

// Computes the 'Intersection over Union' measure between 'box_1' and 'box_2'
float computeIoU(const cv::Rect& box_1, const cv::Rect& box_2);

// Determines the precision and recall measures for a set of ground truth and bounding boxes
void computePrecisionRecall(const std::vector<BoundingBox>& gt_boxes, const std::vector<BoundingBox>& det_boxes, const float iou_threshold, std::vector<float>& precisions, std::vector<float>& recalls)

// Calculates the Single-class Average Precision measure using the 11-point interpolation method
void computeAP(const std::vector<float>& precisions, const std::vector<float>& recalls, float& ap);

// Computes the Multi-class Mean Average Precision
void computeMAP(const std::vector<std::vector<BoundingBox>>& gt_boxes, const std::vector<std::vector<BoundingBox>>& det_boxes, int num_classes, const float iouThreshold, float& map);

void computeIoUBetweenImages(const cv::Mat& groundTruth, const cv::Mat& prediction, int classId, double& iou);

void computeMeanIoU(const cv::Mat& groundTruth, const cv::Mat& prediction, int numClasses, double& miou);

#endif //BILIARDVIDEOANALYSIS_PERFORMANCE_H
