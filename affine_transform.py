import math

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

def get_pixel_safe(img, x, y, c, w, h):
    if 0 <= x < w and 0 <= y < h:
        return img[y][x][c]
    return 0

def warp_affine(src_img, M, dsize):
    dst_w, dst_h = dsize
    src_h = len(src_img)
    src_w = len(src_img[0])
    channels = len(src_img[0][0]) # Number of channels (usually 3)

    # 1. Decompose the transformation matrix (M)
    # [ a  b  tx ]
    # [ c  d  ty ]
    a, b, tx = M[0]
    c, d, ty = M[1]

    # 2. Preparation for inverse matrix calculation (Cramer's Rule)
    # Determinant
    det = a * d - b * c
    
    if det == 0:
        raise ValueError("Determinant is 0, so inverse matrix cannot be calculated.")

    # Pre-calculate the inverse transformation constants (speed optimization)
    # x_src = (d(x_dst - tx) - b(y_dst - ty)) / det
    # y_src = (-c(x_dst - tx) + a(y_dst - ty)) / det
    inv_det = 1.0 / det

    # Create an empty list to store the result image
    dst_img = []

    # 3. Iterate over all pixels (Backward Mapping)
    for dy in range(dst_h):
        row = []
        for dx in range(dst_w):
            
            # (3-1) Convert destination coordinates (dx, dy) to source coordinates (src_x, src_y)
            # To center the coordinates, first subtract the parallel movement (tx, ty).
            rel_x = dx - tx
            rel_y = dy - ty
            
            src_x = (d * rel_x - b * rel_y) * inv_det
            src_y = (-c * rel_x + a * rel_y) * inv_det

            # (3-2) Bilinear Interpolation (Bilinear Interpolation)
            
            # Separate the integer part (x0, y0) and the decimal part (alpha, beta)
            x0 = int(math.floor(src_x))
            y0 = int(math.floor(src_y))
            x1 = x0 + 1
            y1 = y0 + 1

            # Weight (0.0 ~ 1.0)
            alpha = src_x - x0  # x-axis weight
            beta = src_y - y0   # y-axis weight
            
            # Opposite weight
            inv_alpha = 1.0 - alpha
            inv_beta = 1.0 - beta

            pixel_values = []
            
            # Interpolate for each channel (R, G, B)
            for ch in range(channels):
                # Get 4 adjacent pixels (including range check)
                p_a = get_pixel_safe(src_img, x0, y0, ch, src_w, src_h) # Top-Left
                p_b = get_pixel_safe(src_img, x1, y0, ch, src_w, src_h) # Top-Right
                p_c = get_pixel_safe(src_img, x0, y1, ch, src_w, src_h) # Bottom-Left
                p_d = get_pixel_safe(src_img, x1, y1, ch, src_w, src_h) # Bottom-Right

                # Formula: f(x, y) = (1-b) * (Top_Interp) + b * (Bottom_Interp)
                top_interp = (inv_alpha * p_a) + (alpha * p_b)
                bottom_interp = (inv_alpha * p_c) + (alpha * p_d)
                
                final_val = (inv_beta * top_interp) + (beta * bottom_interp)
                
                # Clip and convert to integer
                final_val = int(round(final_val))
                final_val = max(0, min(255, final_val))
                
                pixel_values.append(final_val)
            
            row.append(pixel_values)
        dst_img.append(row)

    return dst_img