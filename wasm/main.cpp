#include <algorithm>
#include <iostream>

#include "geometry.h"
#include "predict.h"
#include "model_weights.h"

constexpr std::array<std::size_t, 5> LEFT_IRIS_INDICES = {468, 469, 470, 471, 472};
constexpr std::array<std::size_t, 5> RIGHT_IRIS_INDICES = {473, 474, 475, 476, 477};

constexpr std::size_t NORMALIZED_IMAGE_SIZE = 256;
constexpr std::size_t EYE_DISTANCE = 96;

extern "C" {
    void process_frame(
        char* rgba_image,
        int image_width,
        int image_height,
        const float* face_landmarks,
        int num_face_landmarks,
        float* output_scores) {

        std::vector<geo::Point> landmark_points;
        landmark_points.reserve(num_face_landmarks / 2);

        for (std::size_t i = 0; i < num_face_landmarks; i += 2) {
            landmark_points.push_back(geo::Point{
                face_landmarks[i] * image_width,
                face_landmarks[i + 1] * image_height
            });
        }

        std::vector<geo::Point> left_iris_points;
        std::vector<geo::Point> right_iris_points;
        left_iris_points.reserve(LEFT_IRIS_INDICES.size());
        right_iris_points.reserve(RIGHT_IRIS_INDICES.size());

        for (const auto& index : LEFT_IRIS_INDICES) {
            left_iris_points.push_back(landmark_points[index]);
        }
        geo::Point left_iris_center = geo::mean(left_iris_points);

        for (const auto& index : RIGHT_IRIS_INDICES) {
            right_iris_points.push_back(landmark_points[index]);
        }

        geo::Point right_iris_center = geo::mean(right_iris_points);

        std::vector<geo::Point> src_points = {
            left_iris_center,
            right_iris_center,
        };

        std::vector<geo::Point> dst_points = {
            {NORMALIZED_IMAGE_SIZE / 2.0 - EYE_DISTANCE / 2.0, 50},
            {NORMALIZED_IMAGE_SIZE / 2.0 + EYE_DISTANCE / 2.0, 50},
        };

        float affine_matrix[2][3];
        bool success = geo::estimateAffinePartial2D(
            src_points,
            dst_points,
            affine_matrix);

        if (!success) {
            // Fill output with zeros if transformation fails
            for (int i = 0; i < OUTPUT_DIM; i++) {
                output_scores[i] = 0.0f;
            }
            return;
        }

        auto normalized_points = geo::transformLandmarks(
            landmark_points,
            affine_matrix);

        float* input = reinterpret_cast<float*>(normalized_points.data());
        float output[OUTPUT_DIM];
        forward(input, output);

        // Copy results to output buffer
        for (int i = 0; i < OUTPUT_DIM; i++) {
            output_scores[i] = output[i];
        }
    }
}
