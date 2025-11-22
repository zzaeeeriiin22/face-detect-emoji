def estimate_affine_partial_2d(src_points, dst_points):
    n = len(src_points)
    src_cx = sum(p[0] for p in src_points) / n
    src_cy = sum(p[1] for p in src_points) / n
    dst_cx = sum(p[0] for p in dst_points) / n
    dst_cy = sum(p[1] for p in dst_points) / n

    num_a = 0.0
    num_b = 0.0
    den = 0.0

    for i in range(n):
        sx = src_points[i][0] - src_cx
        sy = src_points[i][1] - src_cy
        dx = dst_points[i][0] - dst_cx
        dy = dst_points[i][1] - dst_cy

        num_a += sx * dx + sy * dy
        num_b += sx * dy - sy * dx
        den   += sx * sx + sy * sy

    if den == 0:
        return None

    a = num_a / den
    b = num_b / den

    tx = dst_cx - (a * src_cx - b * src_cy)
    ty = dst_cy - (b * src_cx + a * src_cy)

    return [[a, -b, tx], [b, a, ty]]
