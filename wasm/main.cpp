#include <algorithm>

#include "emoji.h"
#include "geometry.h"
#include "predict.h"
#include "model_weights.h"

constexpr std::array<std::size_t, 5> LEFT_IRIS_INDICES = {468, 469, 470, 471, 472};
constexpr std::array<std::size_t, 5> RIGHT_IRIS_INDICES = {473, 474, 475, 476, 477};
constexpr std::size_t LOWER_LIP_CENTER_INDEX = 17;
constexpr std::size_t LEFT_FACE_EDGE_INDEX = 127;
constexpr std::size_t RIGHT_FACE_EDGE_INDEX = 356;

constexpr std::size_t NORMALIZED_IMAGE_SIZE = 256;
constexpr std::size_t EYE_DISTANCE = 96;

constexpr std::size_t NUM_FACE_LANDMARKS = 478;

extern "C" {
    void process_frame(
        unsigned char* rgba_image,
        int image_width,
        int image_height,
        const float* face_landmarks,
        int num_faces,
        unsigned int around_face_count) {

        for (int face_index = 0; face_index < num_faces; face_index++) {
            const float* one_face_landmarks = &face_landmarks[face_index * NUM_FACE_LANDMARKS * 2];

            std::vector<geo::Point> landmark_points;
            landmark_points.reserve(NUM_FACE_LANDMARKS);
            
            for (std::size_t i = 0; i < NUM_FACE_LANDMARKS; i++) {
                landmark_points.push_back(geo::Point{
                    one_face_landmarks[i * 2] * image_width,
                    one_face_landmarks[i * 2 + 1] * image_height
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

            float partial_affine_matrix[2][3];
            bool success = geo::estimateAffinePartial2D(
                src_points,
                dst_points,
                partial_affine_matrix);

            if (!success) {
                return;
            }

            auto normalized_points = geo::transformLandmarks(
                landmark_points,
                partial_affine_matrix);

            float* input = reinterpret_cast<float*>(normalized_points.data());
            float output[OUTPUT_DIM];
            forward(input, output);

            int max_index = std::max_element(output, output + OUTPUT_DIM) - output;

            const unsigned char* emoji[] = {
                emoji::ANGRY_EMOJI,
                emoji::HAPPY_EMOJI,
                emoji::NEUTRAL_EMOJI,
                emoji::SURPRISE_EMOJI };

            const std::vector<geo::Point>* emoji_points[] = {
                &emoji::ANGRY_POINTS,
                &emoji::HAPPY_POINTS,
                &emoji::NEUTRAL_POINTS,
                &emoji::SURPRISE_POINTS
            };

            if (around_face_count > 0) {
                float face_width = geo::getDistance(
                    landmark_points[LEFT_FACE_EDGE_INDEX],
                    landmark_points[RIGHT_FACE_EDGE_INDEX]);

                face_width *= 1.2;

                for (int i = 0; i < around_face_count; i++) {
                    std::vector<geo::Point> canvas_points{
                        left_iris_center,
                        right_iris_center,
                        landmark_points[LOWER_LIP_CENTER_INDEX],
                    };

                    for (auto& point : canvas_points) {
                        point.x += std::cos(i * 2 * M_PI / around_face_count) * face_width;
                        point.y += std::sin(i * 2 * M_PI / around_face_count) * face_width;
                    }

                    geo::overlayWarpAffine(
                        rgba_image,
                        image_width,
                        image_height,
                        emoji[max_index],
                        emoji::EMOJI_WIDTH,
                        emoji::EMOJI_HEIGHT,
                        canvas_points,
                        *emoji_points[max_index]);
                }
            }
            else {
                std::vector<geo::Point> canvas_points = {
                    left_iris_center,
                    right_iris_center,
                    landmark_points[LOWER_LIP_CENTER_INDEX],
                };

                geo::overlayWarpAffine(
                    rgba_image,
                    image_width,
                    image_height,
                    emoji[max_index],
                    emoji::EMOJI_WIDTH,
                    emoji::EMOJI_HEIGHT,
                    canvas_points,
                    *emoji_points[max_index]);
            }
        }
    }
}
