#include "bounding_box.h"

ballBoundingBox::ballBoundingBox(cv::Rect &bbox, int id)
    : bbox(bbox), id(id){}

void createBoundingBoxes(const std::vector<billiardBall>& balls, std::vector<ballBoundingBox>& bounding_boxes) {
    for (const billiardBall& ball: balls) {
        int x_corner = ball.x - static_cast<int>(ball.true_radius);
        int y_corner = ball.y - static_cast<int>(ball.true_radius);
        cv::Rect box = cv::Rect (x_corner, y_corner, 2 * static_cast<int>(ball.true_radius), 2 * static_cast<int>(ball.true_radius));
        bounding_boxes.push_back(ballBoundingBox(box,ball.id));
    }
}

