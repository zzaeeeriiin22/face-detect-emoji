#pragma once

#include <vector>

namespace geo {

struct Point {
    float x;
    float y;
};

static_assert(sizeof(Point) == 8, "Point must be 8 bytes");

/* 
 * Calculates the mean of a set of 2D points.
 *
 * @param pts: Input vector of Point structures representing the points.
 * @return A Point structure representing the mean of the points.
 */
Point mean(const std::vector<Point>& pts);

/* 
 * Applies a 2x3 affine transformation matrix to a set of 2D points.
 *
 * @param landmarks: Input vector of Point structures representing the original landmarks.
 * @param matrix_2x3: A 2x3 affine transformation matrix.
 * @return A vector of Point structures representing the transformed landmarks.
 */
std::vector<Point> transformLandmarks(
    const std::vector<Point>& landmarks,
    const float matrix_2x3[][3]
);

/* 
 * Estimates a 2D affine transformation matrix (with rotation, uniform scaling, and translation)
 * that maps src_points to dst_points.
 * Warning: This algorithm does not handle outliers.
 *
 * @param src_points: Vector of Point structures representing the source points.
 * @param dst_points: Vector of Point structures representing the destination points.
 * @param matrix_2x3: Output parameter to hold the estimated 2x3 affine transformation matrix.
 * @return true if the estimation was successful, false otherwise.
 */
bool estimateAffinePartial2D(
    const std::vector<Point>& src_points,
    const std::vector<Point>& dst_points,
    float matrix_2x3[][3]
);

} // namespace geometry
