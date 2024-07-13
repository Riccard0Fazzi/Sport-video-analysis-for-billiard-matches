#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <iostream>


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


int predict(const Ptr<ml::SVM>& svm, const Mat& image) {
    Mat features = extractFeatures(image);
    return svm->predict(features);
}



int main() {
    // Load the trained SVM model
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load("svm_trained_model.yml");

    // Load test images
    std::vector<cv::Mat> test_images;  // Load your test images

    // Predict the class for each test image
    for (const auto& test_image : test_images) {
        std::vector<double> test_features = extract_features(test_image);

        // Convert features to a cv::Mat
        cv::Mat sample(1, test_features.size(), CV_32F);
        for (size_t i = 0; i < test_features.size(); ++i) {
            sample.at<float>(0, i) = static_cast<float>(test_features[i]);
        }

        // Predict the class
        int prediction = svm->predict(sample);

        // Output the prediction
        switch (prediction) {
            case 0:
                std::cout << "Prediction: Full Ball" << std::endl;
                break;
            case 1:
                std::cout << "Prediction: Half Ball" << std::endl;
                break;
            case 2:
                std::cout << "Prediction: White Ball" << std::endl;
                break;
            case 3:
                std::cout << "Prediction: Black Ball" << std::endl;
                break;
		    case 4:
                std::cout << "Prediction: No Ball" << std::endl;
                break;

        }
    }

    return 0;
}

