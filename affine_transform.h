#pragma once

#include <vector>
#include "image_types.h"

bool estimateAffinePartial2D(
    const std::vector<Point>& src_points,
    const std::vector<Point>& dst_points,
    double matrix_2x3[][3]
);