#include <vector>
#include "image_types.h"

std::vector<Point> transformLandmarks(
    const std::vector<Point>& landmarks,
    const double matrix_2x3[2][3]
) {
    std::vector<Point> transformed;
    transformed.reserve(landmarks.size());

    for (const Point& pt : landmarks) {
        Point new_pt;
        new_pt.x = matrix_2x3[0][0] * pt.x + matrix_2x3[0][1] * pt.y + matrix_2x3[0][2];
        new_pt.y = matrix_2x3[1][0] * pt.x + matrix_2x3[1][1] * pt.y + matrix_2x3[1][2];
        transformed.push_back(new_pt);
    }

    return transformed;
}
