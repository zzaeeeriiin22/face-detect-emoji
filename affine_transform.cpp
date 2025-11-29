#include <limits>
#include <numeric>

#include "image_types.h"
#include "affine_transform.h"

std::tuple<double, double> meanXY(const std::vector<Point>& pts) {
    if(pts.empty()) return {0.0, 0.0};

    double sum_x = 0.0;
    double sum_y = 0.0;

    for (const auto& p : pts) {
        sum_x += p.x;
        sum_y += p.y;
    }

    double mean_x = sum_x / static_cast<double>(pts.size());
    double mean_y = sum_y / static_cast<double>(pts.size());

    return {mean_x, mean_y};
}

bool estimateAffinePartial2D(
    const std::vector<Point>& src_points,
    const std::vector<Point>& dst_points,
    double matrix_2x3[][3]
) {
    int n = src_points.size();

    auto [src_cx,src_cy] = meanXY(src_points);
    auto [dst_cx, dst_cy] = meanXY(dst_points);

    double num_a = 0.0;
    double num_b = 0.0;
    double den = 0.0;

    for (int i = 0; i < n; i++) {
        // 표준화
        double sx = src_points[i].x - src_cx;
        double sy = src_points[i].y - src_cy;
        double dx = dst_points[i].x - dst_cx;
        double dy = dst_points[i].y - dst_cy;

        // 회전 및 스케일 추정
        // cos 선분 (vector 내적)
        //두 좌표가 얼마나 같은 방향을 보고 있는가 
        num_a += sx * dx + sy * dy; 
        
        // sin 선분 (vector 외적)
        // 두 좌표가 얼마나 서로 직교 하는가 (90도일 때 가장 커짐)
        num_b += sx * dy - sy * dx; 

        // a,b의 대표값을 찾기 위함
        den += sx * sx + sy * sy;
    }

    if (den == 0.0) {
        return false;
    }

    double a = num_a / den;
    double b = num_b / den;

    // 평행 이동
    // (a * src_cx - b * src_cy): 위에서 구한 a,b로 회전, 스케일 적용
    double tx = dst_cx - (a * src_cx - b * src_cy);
    double ty = dst_cy - (b * src_cx + a * src_cy);

    matrix_2x3[0][0] = a; 
    matrix_2x3[0][1] = -b;
    matrix_2x3[0][2] = tx;
    matrix_2x3[1][0] = b;
    matrix_2x3[1][1] = a;
    matrix_2x3[1][2] = ty;

    return true;
}