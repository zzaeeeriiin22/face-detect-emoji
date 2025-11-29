#include <cassert>
#include <tuple>
#include "geometry.h"

namespace geo {

Point mean(const std::vector<Point>& pts) {
    if (pts.empty()) {
        return {0.0, 0.0};
    }

    float sum_x = 0.0;
    float sum_y = 0.0;

    for (const auto& p : pts) {
        sum_x += p.x;
        sum_y += p.y;
    }

    float mean_x = sum_x / static_cast<float>(pts.size());
    float mean_y = sum_y / static_cast<float>(pts.size());

    return Point{mean_x, mean_y};
}

std::vector<Point> transformLandmarks(
    const std::vector<Point>& landmarks,
    const float matrix_2x3[][3]
) {
    assert(matrix_2x3 != nullptr);

    std::vector<Point> transformed;

    // 미리 메모리 할당
    transformed.reserve(landmarks.size());

    for (const Point& pt : landmarks) {
        Point new_pt;
        // Affine transformation 적용
        new_pt.x = \
            matrix_2x3[0][0] * pt.x \
            + matrix_2x3[0][1] * pt.y \
            + matrix_2x3[0][2];

        new_pt.y = \
            matrix_2x3[1][0] * pt.x \
            + matrix_2x3[1][1] * pt.y \
            + matrix_2x3[1][2];

        transformed.push_back(new_pt);
    }
    return transformed;
}

bool estimateAffinePartial2D(
    const std::vector<Point>& src_points,
    const std::vector<Point>& dst_points,
    float matrix_2x3[][3]
) {
    assert(src_points.size() == dst_points.size());
    assert(src_points.size() >= 2);

    int n = src_points.size();

    Point src_center = mean(src_points);
    Point dst_center = mean(dst_points);

    float num_a = 0.0;
    float num_b = 0.0;
    float den = 0.0;

    for (int i = 0; i < n; i++) {
        // 표준화
        float sx = src_points[i].x - src_center.x;
        float sy = src_points[i].y - src_center.y;
        float dx = dst_points[i].x - dst_center.x;
        float dy = dst_points[i].y - dst_center.y;

        // 회전 및 스케일 추정
        // cos 선분 (vector 내적)
        // 두 좌표가 얼마나 같은 방향을 보고 있는가
        num_a += sx * dx + sy * dy;

        // sin 선분 (vector 외적)
        // 두 좌표가 얼마나 서로 직교 하는가 (90도일 때 가장 커짐)
        num_b += sx * dy - sy * dx;

        // a, b 의 분모 계산을 위한 제곱합
        den += sx * sx + sy * sy;
    }

    if (den == 0.0) {
        return false;
    }

    float a = num_a / den;
    float b = num_b / den;

    // 평행 이동
    // (a * src_center.x - b * src_center.y): 위에서 구한 a,b로 회전, 스케일 적용
    float tx = dst_center.x - (a * src_center.x - b * src_center.y);
    float ty = dst_center.y - (b * src_center.x + a * src_center.y);

    matrix_2x3[0][0] = a;
    matrix_2x3[0][1] = -b;
    matrix_2x3[0][2] = tx;
    matrix_2x3[1][0] = b;
    matrix_2x3[1][1] = a;
    matrix_2x3[1][2] = ty;

    return true;
}

} // namespace geometry
