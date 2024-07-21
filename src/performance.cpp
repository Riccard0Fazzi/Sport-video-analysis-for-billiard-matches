// Created by Tommaso Tubaldo on 18/07/24 - Hours:

#include "../include/performance.h"

void readImagesAndBBoxes(const std::string& folder_path, std::vector<cv::Mat>& frames, std::vector<cv::Mat>& masks, std::vector<std::vector<BoundingBox>>& bounding_boxes) {
    // Paths to the inner folders
    std::string frames_path = folder_path + "/frames";
    std::string masks_path = folder_path + "/masks";
    std::string bounding_boxes_path = folder_path + "/bounding_boxes";

    // For every image in the folder, it reads each frame and allocates it in the frames vector
    for (const auto& entry : std::__fs::filesystem::directory_iterator(frames_path)) {
        if (entry.is_regular_file()) {
            frames.push_back(cv::imread(entry.path().string(),cv::IMREAD_COLOR));
        }
    }

    // For every image in the folder, it reads each mask and allocates it in the masks vector
    for (const auto& entry : std::__fs::filesystem::directory_iterator(masks_path)) {
        if (entry.is_regular_file()) {
            masks.push_back(cv::imread(entry.path().string(),cv::IMREAD_GRAYSCALE));
        }
    }

    // Iterate for each text file
    for (const auto& entry : std::__fs::filesystem::directory_iterator(bounding_boxes_path)) {
        // Checks if the folder was accessed correctly
        if (entry.is_regular_file()) {
            std::vector<BoundingBox> bounding_boxes_for_image;
            std::ifstream file(entry.path());
            // Checks if the file was opened correctly
            if (file.is_open()) {
                std::string line;
                while (std::getline(file, line)) {
                    std::istringstream iss(line);
                    int x, y, width, height, class_id;
                    // If the row is correctly defined, it allocates each element in the BoundingBox structure
                    if (iss >> x >> y >> width >> height >> class_id) {
                        BoundingBox bbox;
                        bbox.box = cv::Rect(x, y, width, height);
                        bbox.class_id = class_id;
                        bounding_boxes_for_image.push_back(bbox);
                    }
                }
            }
            bounding_boxes.push_back(bounding_boxes_for_image);
        }
    }
}

void generateGroundTruth(const std::string& folder_path, std::vector<cv::Mat>& all_frames, std::vector<cv::Mat>& all_masks, std::vector<std::vector<BoundingBox>>& all_bounding_boxes) {
    // Iterate through the 10 main folders
    for (const auto& entry : std::__fs::filesystem::directory_iterator(folder_path)) {
        if (entry.is_directory()) {
            // Every frame, mask and bbox is pushed back on the "all" vectors
            readImagesAndBBoxes(entry.path().string(),all_frames,all_masks,all_bounding_boxes);
        }
    }
}

float computeIoUBetweenBBoxes(const cv::Rect& box_1, const cv::Rect& box_2) {
    int x_1 = std::max(box_1.x, box_2.x);
    int y_1 = std::max(box_1.y, box_2.y);
    int x_2 = std::min(box_1.x + box_1.width, box_2.x + box_2.width);
    int y_2 = std::min(box_1.y + box_1.height, box_2.y + box_2.height);

    // Computes, respectively, interference, box_1 and box_2 areas
    int interArea = std::max(0, x_2 - x_1 + 1) * std::max(0, y_1 - y_2 + 1);
    int box_1_area = box_1.width * box_1.height;
    int box_2_area = box_2.width * box_2.height;

    return float(interArea) / float(box_1_area + box_2_area - interArea);
}

void computePrecisionRecall(const std::vector<BoundingBox>& gt_boxes, const std::vector<BoundingBox>& det_boxes, const float iou_threshold, std::vector<float>& precisions, std::vector<float>& recalls) {
    // Initialize the boolean vectors for ground truth and detected boxes.
    // If a ground truth box is been matched, then 'gt_matched[i] = true'; the same for det_matched
    std::vector<bool> gt_matched(gt_boxes.size(), false);
    std::vector<bool> det_matched(det_boxes.size(), false);

    int tp = 0, fp = 0, fn = gt_boxes.size();
    float iou;

    // Iterate for each detected box
    for (size_t i = 0; i < det_boxes.size(); i++) {
        float maxIoU = 0.0;
        int bestMatch = -1;

        for (size_t j = 0; j < gt_boxes.size(); j++) {
            // If a ground truth box has already been matched or its class id
            // does not match the detected box's class id, it is skipped
            if (gt_matched[j] || det_boxes[i].class_id != gt_boxes[j].class_id)
                continue;

            iou = computeIoU(det_boxes[i].box, gt_boxes[j].box);
            //The best matching between detected and gt boxes is selected
            if (iou > maxIoU) {
                maxIoU = iou;
                bestMatch = j;
            }
        }

        // If the IoU of the best match is above the threshold, then it is a true positive
        if (maxIoU >= iou_threshold) {
            gt_matched[bestMatch] = true;
            det_matched[i] = true;
            tp++;
            fn--;
        } else {
            fp++;
        }

        // Compute the precision and recall values and store them
        precisions.push_back(tp / float(tp + fp));
        recalls.push_back(tp / float(tp + fn));
    }
}

void computeAP(const std::vector<float>& precisions, const std::vector<float>& recalls, float& ap) {
    ap = 0.0;
    // Iterate for 11 times, as the points needed for interpolation
    for (float t = 0.0; t <= 1.0; t += 0.1) {
        float p = 0.0;
        // Determine the interpolated precision, which is the maximum precision
        // corresponding to the recall value greater than the current recall value.
        for (size_t i = 0; i < precisions.size(); i++) {
            if (recalls[i] >= t) {
                p = std::max(p, precisions[i]);
            }
        }
        ap += p;
    }
    ap / 11.0;
}

void computeMAP(const std::vector<std::vector<BoundingBox>>& gt_boxes, const std::vector<std::vector<BoundingBox>>& det_boxes, int num_classes, const float iouThreshold, float& map) {
    map = 0.0;
    // Iterate over each class
    for (int c = 0; c < num_classes; c++) {
        // Extract each box belonging to the class involved
        std::vector<BoundingBox> gt_class_boxes, det_class_boxes;
        for (const auto& boxes : gt_boxes) {
            for (const auto& box : boxes) {
                if (box.class_id == c)
                    gt_class_boxes.push_back(box);
            }
        }
        for (const auto& boxes : det_boxes) {
            for (const auto& box : boxes) {
                if (box.class_id == c)
                    det_class_boxes.push_back(box);
            }
        }

        // Compute precision and recall for each detected box
        std::vector<float> precisions, recalls;
        computePrecisionRecall(gt_class_boxes, det_class_boxes, iouThreshold, precisions, recalls);

        // Compute average precision and increment map
        float ap;
        computeAP(precisions, recalls, ap);
        map += ap;
    }
    // Determine the mean of the Average Precision
    map / num_classes;
}

void computeIoUBetweenImages(const cv::Mat& groundTruth, const cv::Mat& prediction, int classId, double& iou) {
    // Create binary masks for the current class
    cv::Mat gtMask = (groundTruth == classId);
    cv::Mat predMask = (prediction == classId);

    // Compute intersection and union
    cv::Mat intersection, unionMat;
    cv::bitwise_and(gtMask, predMask, intersection);
    cv::bitwise_or(gtMask, predMask, unionMat);

    double intersectionArea = cv::countNonZero(intersection);
    double unionArea = cv::countNonZero(unionMat);

    if (unionArea == 0) {
        iou = (intersectionArea == 0) ? 1.0 : 0.0; // If both are zero, IoU is 1, else 0
    }

    iou = intersectionArea / unionArea;
}

void computeMeanIoU(const std::vector<cv::Mat>& groundTruth, const std::vector<cv::Mat>& segmented_images, int numClasses, std::vector<double> class_iou, double& miou) {
    std::vector<double> iou_temp;

    // Iterate over each class and compute the IoU, which is stored in 'class_iou'
    for (int classId = 0; classId < numClasses; classId++) {
        // For each image the IoU of one class is computed by means of an average on all the IoU of every image
        for (int i = 0; i < groundTruth.size(); i++) {
            double iou;
            computeIoUBetweenImages(groundTruth[i], segmented_images[i], classId, iou);
            iou_temp.push_back(iou);
        }
        class_iou.push_back(std::accumulate(iou_temp.begin(),iou_temp.end(),0.0) / groundTruth.size());
        iou_temp.clear();
    }

    // Calculates the average on the overall IoU of each class
    miou = std::accumulate(class_iou.begin(),class_iou.end(),0.0) / numClasses;
}
