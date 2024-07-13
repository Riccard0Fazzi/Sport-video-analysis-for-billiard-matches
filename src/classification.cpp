// Created by Tommaso Tubaldo on 04/07/24.
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

using namespace cv;

Mat extractFeatures(const Mat& image);
void loadData(std::vector<Mat>& images, std::vector<int>& labels);
Ptr<ml::SVM> trainSVM(const std::vector<Mat>& trainData, const std::vector<int>& trainLabels);
int predict(const Ptr<ml::SVM>& svm, const Mat& image);

int main(int argc, char** argv) {

}

Mat extractFeatures(const Mat& image) {
    // Resize image to a fixed size
    Mat resized_image;
    resize(image,resized_image,Size(64,64));

    // Convert to HSV color space
    Mat hsvImage;
    cvtColor(resized_image,hsvImage,COLOR_BGR2HSV);

    // Calculate color histogram
    int hBins = 50, sBins = 60;
    int histSize[] = {hBins, sBins};
    float hRanges[] = {0, 180};
    float sRanges[] = {0, 256};
    const float* ranges[] = {hRanges, sRanges};
    MatND hist;
    int channels[] = {0, 1};
    calcHist(&hsvImage, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());

    // Convert histogram to a single row feature vector
    return hist.reshape(1, 1);
}

// Function to save features and labels to CSV
void saveToCSV(const std::string& filename, const std::vector<Mat>& features, const std::vector<int>& labels) {
    std::ofstream file(filename);

    if (file.is_open()) {
        for (size_t i = 0; i < features.size(); ++i) {
            Mat row = features[i];
            for (int j = 0; j < row.cols; ++j) {
                file << row.at<float>(0, j);
                if (j < row.cols - 1) {
                    file << ",";
                }
            }
            file << "," << labels[i] << "\n";
        }
        file.close();
    } else {
        std::cout << "Unable to open file for writing: " << filename << std::endl;
    }
}

void loadData(std::vector<Mat>& images, std::vector<int>& labels) {
    // Load images and labels
    // For simplicity, assume images are stored in directories named after their labels
    std::vector<std::string> categories = {"white", "black", "solid", "striped"};
    for (int i = 0; i < categories.size(); ++i) {
        std::string category = categories[i];
        std::vector<std::string> filenames;
        glob("path/to/data/" + category + "/*.jpg", filenames);
        for (size_t j = 0; j < filenames.size(); ++j) {
            Mat image = imread(filenames[j]);
            if (!image.empty()) {
                images.push_back(extractFeatures(image));
                labels.push_back(i);
            }
        }
    }
}

// Function to load features and labels from CSV
void loadFromCSV(const std::string& filename, std::vector<Mat>& features, std::vector<int>& labels) {
    std::ifstream file(filename);
    std::string line, cell;

    if (file.is_open()) {
        while (getline(file, line)) {
            std::stringstream lineStream(line);
            std::vector<float> featureRow;
            while (getline(lineStream, cell, ',')) {
                featureRow.push_back(stof(cell));
            }
            int label = static_cast<int>(featureRow.back());
            featureRow.pop_back();
            Mat featureMat(1, featureRow.size(), CV_32F, featureRow.data());
            features.push_back(featureMat.clone());
            labels.push_back(label);
        }
        file.close();
    } else {
        std::cout << "Unable to open file for reading: " << filename << std::endl;
    }
}

Ptr<ml::SVM> trainSVM(const std::vector<Mat>& trainData, const std::vector<int>& trainLabels) {
    // Convert data to the right format
    Mat trainingData;
    vconcat(trainData, trainingData);
    Mat labelsMat(trainLabels.size(), 1, CV_32S);
    for (size_t i = 0; i < trainLabels.size(); ++i) {
        labelsMat.at<int>(i, 0) = trainLabels[i];
    }

    // Train SVM
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(trainingData, ml::ROW_SAMPLE, labelsMat);
    return svm;
}

int predict(const Ptr<ml::SVM>& svm, const Mat& image) {
    Mat features = extractFeatures(image);
    return svm->predict(features);
}