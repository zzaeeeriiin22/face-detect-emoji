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


float getDeterminant3x3(float a, float b, float c,
                          float d, float e, float f,
                          float g, float h, float i) {
    return (a * (e * i - f * h) -
            b * (d * i - f * g) +
            c * (d * h - e * g));
}

bool estimateAffineFull2D(
    const std::vector<Point>& pts_in,
    const std::vector<Point>& pts_out,
    float matrix_2x3[][3]
) {
    assert(pts_in.size() == pts_out.size());
    assert(pts_in.size() == 3);

    float x1 = pts_in[0].x, y1 = pts_in[0].y;
    float x2 = pts_in[1].x, y2 = pts_in[1].y;
    float x3 = pts_in[2].x, y3 = pts_in[2].y;

    float u1 = pts_out[0].x, v1 = pts_out[0].y;
    float u2 = pts_out[1].x, v2 = pts_out[1].y;
    float u3 = pts_out[2].x, v3 = pts_out[2].y;

    float det_A = getDeterminant3x3(
        x1, y1, 1,
        x2, y2, 1,
        x3, y3, 1);

    if (std::abs(det_A) < 1e-9) {
        return false;  // 계산 불가 (일직선 등)
    }

    // a, b, c 계산
    float det_a = getDeterminant3x3(
        u1, y1, 1,
        u2, y2, 1,
        u3, y3, 1);

    float det_b = getDeterminant3x3(
        x1, u1, 1,
        x2, u2, 1,
        x3, u3, 1);

    float det_c = getDeterminant3x3(
        x1, y1, u1,
        x2, y2, u2,
        x3, y3, u3);

    // d, e, f 계산
    float det_d = getDeterminant3x3(
        v1, y1, 1,
        v2, y2, 1,
        v3, y3, 1);

    float det_e = getDeterminant3x3(
        x1, v1, 1,
        x2, v2, 1,
        x3, v3, 1);

    float det_f = getDeterminant3x3(
        x1, y1, v1,
        x2, y2, v2,
        x3, y3, v3);

    float a = det_a / det_A;
    float b = det_b / det_A;
    float c = det_c / det_A;
    float d = det_d / det_A;
    float e = det_e / det_A;
    float f = det_f / det_A;

    matrix_2x3[0][0] = a;
    matrix_2x3[0][1] = b;
    matrix_2x3[0][2] = c;
    matrix_2x3[1][0] = d;
    matrix_2x3[1][1] = e;
    matrix_2x3[1][2] = f;

    return true;
}

inline Point applyTransform(Point pt, const float matrix_2x3[][3]) {
    Point new_pt;
    new_pt.x = matrix_2x3[0][0] * pt.x + matrix_2x3[0][1] * pt.y + matrix_2x3[0][2];
    new_pt.y = matrix_2x3[1][0] * pt.x + matrix_2x3[1][1] * pt.y + matrix_2x3[1][2];
    return new_pt;
}

bool overlayWarpAffine(
    unsigned char* canvas_data, int canvas_width, int canvas_height,
    const unsigned char* src_data, int src_width, int src_height,
    const std::vector<Point>& pts_canvas,
    const std::vector<Point>& pts_src) {

    assert(canvas_data != nullptr);
    assert(src_data != nullptr);
    assert(pts_canvas.size() == 3);
    assert(pts_src.size() == 3);

    // 1. 행렬 계산
    float forward_matrix[2][3];
    if (!estimateAffineFull2D(pts_src, pts_canvas, forward_matrix)) return false;

    float inverse_matrix[2][3];
    if (!estimateAffineFull2D(pts_canvas, pts_src, inverse_matrix)) return false;

    // 2. Bounding Box 계산
    std::vector<Point> corners = {
        {0.0f, 0.0f},
        {(float)src_width, 0.0f},
        {(float)src_width, (float)src_height},
        {0.0f, (float)src_height}
    };

    float min_x = 1e9f, max_x = -1e9f;
    float min_y = 1e9f, max_y = -1e9f;

    for (const auto& pt : corners) {
        Point tf = applyTransform(pt, forward_matrix);
        if (tf.x < min_x) min_x = tf.x;
        if (tf.x > max_x) max_x = tf.x;
        if (tf.y < min_y) min_y = tf.y;
        if (tf.y > max_y) max_y = tf.y;
    }

    int start_x = std::max(0, (int)std::floor(min_x));
    int end_x   = std::min(canvas_width, (int)std::ceil(max_x));
    int start_y = std::max(0, (int)std::floor(min_y));
    int end_y   = std::min(canvas_height, (int)std::ceil(max_y));

    float a = inverse_matrix[0][0], b = inverse_matrix[0][1], c = inverse_matrix[0][2];
    float d = inverse_matrix[1][0], e = inverse_matrix[1][1], f = inverse_matrix[1][2];

    const int CHANNELS = 4; // RGBA

    // 3. 픽셀 순회 및 bilinear 합성
    for (int y = start_y; y < end_y; y++) {
        unsigned char* dst_row = canvas_data + (y * canvas_width * CHANNELS);

        for (int x = start_x; x < end_x; x++) {
            float src_xf = a * x + b * y + c;
            float src_yf = d * x + e * y + f;

            int src_x0 = (int)std::floor(src_xf);
            int src_y0 = (int)std::floor(src_yf);
            float fx = src_xf - src_x0;
            float fy = src_yf - src_y0;
            int src_x1 = src_x0 + 1;
            int src_y1 = src_y0 + 1;

            if (
                src_x0 >= 0 && src_x1 < src_width &&
                src_y0 >= 0 && src_y1 < src_height
            ) {
                float rgba[4] = {0, 0, 0, 0};
                for (int cix = 0; cix < 4; cix++) {
                    // (0,0)
                    const unsigned char v00 = src_data[(src_y0 * src_width + src_x0) * CHANNELS + cix];
                    // (1,0)
                    const unsigned char v10 = src_data[(src_y0 * src_width + src_x1) * CHANNELS + cix];
                    // (0,1)
                    const unsigned char v01 = src_data[(src_y1 * src_width + src_x0) * CHANNELS + cix];
                    // (1,1)
                    const unsigned char v11 = src_data[(src_y1 * src_width + src_x1) * CHANNELS + cix];

                    float val =
                        (1 - fx) * (1 - fy) * v00 +
                        fx       * (1 - fy) * v10 +
                        (1 - fx) * fy       * v01 +
                        fx       * fy       * v11;
                    rgba[cix] = val;
                }

                // Alpha blending (paste source pixel "over" destination pixel)
                int dst_col_idx = x * CHANNELS;
                float alpha = rgba[3] / 255.0f;
                if (alpha > 0.0f) {
                    for (int cix = 0; cix < 3; cix++) {
                        dst_row[dst_col_idx + cix] = 
                            (unsigned char)std::round(rgba[cix] * alpha + dst_row[dst_col_idx + cix] * (1.0f - alpha));
                    }
                    // Update alpha
                    dst_row[dst_col_idx + 3] = 
                        (unsigned char)std::round(255.0f * (alpha + (dst_row[dst_col_idx + 3] / 255.0f) * (1.0f - alpha)));
                }
            }
        }
    }
    return true;
}

float getDistance(const Point& p1, const Point& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

}  // namespace geo
