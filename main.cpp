#include <iostream>
#include <iomanip>
#include <vector>

#include "image_types.h"
#include "affine_transform.h"

int main() {
    // 1. 더미 포인트 데이터 설정
    std::vector<Point> src_points = {
        {241.5497589111328, 303.3517761230469},
        {402.2906799316406, 302.87603759765625}
    };

    std::vector<Point> dst_points = {
        {80.0, 50.0},
        {176.0, 50.0}
    };

    // 2. 어파인 추정 실행
    double matrix_2x3[2][3];
    bool ok = estimateAffinePartial2D(src_points, dst_points, matrix_2x3);

    if (!ok) {
        std::cerr << "estimateAffinePartial2D failed (den == 0 or invalid input)\n";
        return 1;
    }

    // 3. 결과 출력 (네가 가진 기대 행렬과 비교)
    std::cout << std::fixed << std::setprecision(15);

    // std::cout << "=== Estimated Affine Matrix ===\n";
    // std::cout << "[ " << M.m00 << ", " << M.m01 << ", " << M.m02 << " ]\n";
    // std::cout << "[ " << M.m10 << ", " << M.m11 << ", " << M.m12 << " ]\n\n";

    // AffineTransform expected {
    //     0.5972291217278474, -0.001767595332210985, -63.72434718457612,
    //     0.001767595332210985,  0.5972291217278474, -131.59747705489787
    // };

    double expected[2][3] = {
        {0.5972291217278474, -0.001767595332210985, -63.72434718457612},
        {0.001767595332210985,  0.5972291217278474, -131.59747705489787}
    };

    std::cout << "=== Expected Affine Matrix ===\n";
    std::cout << "[ " << expected[0][0] << ", " << expected[0][1] << ", " << expected[0][2] << " ]\n";
    std::cout << "[ " << expected[1][0] << ", " << expected << " ]\n";

    std::cout << "=== Difference (Estimated - Expected) ===\n";
    std::cout << "[ "
              << (matrix_2x3[0][0] - expected[0][0]) << ", "
              << (matrix_2x3[0][1] - expected[0][1]) << ", "
              << (matrix_2x3[0][2] - expected[0][2]) << " ]\n";
    std::cout << "[ "
              << (matrix_2x3[1][0] - expected[1][0]) << ", "
              << (matrix_2x3[1][1] - expected[1][1]) << ", "
              << (matrix_2x3[1][2] - expected[1][2]) << " ]\n\n";
    
    // 4. warpAffine 더미 테스트 (간단한 그라디언트 이미지)
    // const int src_w = 256;
    // const int src_h = 256;
    // const int channels = 3;

    // Image src_img(src_w, src_h, channels);

    // // R: x 그라디언트, G: y 그라디언트, B: 128 고정
    // for (int y = 0; y < src_h; ++y) {
    //     for (int x = 0; x < src_w; ++x) {
    //         src_img.setPixel(x, y, 0, static_cast<uint8_t>(x % 256)); // R
    //         src_img.setPixel(x, y, 1, static_cast<uint8_t>(y % 256)); // G
    //         src_img.setPixel(x, y, 2, 128);                           // B
    //     }
    // }

    // // 추정된 행렬로 워핑 (사이즈는 그대로 256x256)
    // Image dst_img = warpAffine(src_img, M, src_w, src_h);

    // // 몇 개 픽셀만 콘솔에 찍어서 정상 동작 확인
    // std::cout << "Sample dst pixels:\n";
    // for (int y = 0; y <= 200; y += 50) {
    //     for (int x = 0; x <= 200; x += 50) {
    //         int idx = (y * dst_img.width + x) * dst_img.channels;
    //         uint8_t r = dst_img.data[idx + 0];
    //         uint8_t g = dst_img.data[idx + 1];
    //         uint8_t b = dst_img.data[idx + 2];

    //         std::cout << "dst(" << x << ", " << y << ") = ("
    //                   << static_cast<int>(r) << ", "
    //                   << static_cast<int>(g) << ", "
    //                   << static_cast<int>(b) << ")\n";
    //     }
    // }

    return 0;
}
