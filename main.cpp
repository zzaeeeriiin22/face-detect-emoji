#include <iostream>
#include <opencv2/opencv.hpp>

#include "muller_landmarks.h"

int main() {
  std::cout << "OpenCV Image Display Demo" << std::endl;
  std::cout << "OpenCV version: " << CV_VERSION << std::endl;

  // 이미지 읽기
  cv::Mat image = cv::imread("muller.webp");

  // 이미지 로드 확인
  if (image.empty()) {
    std::cerr << "Error: Could not load image 'muller.webp'" << std::endl;
    std::cerr
        << "Make sure muller.webp is in the same directory as the executable"
        << std::endl;
    return -1;
  }

  std::cout << "Image loaded successfully!" << std::endl;
  std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;
  std::cout << "Press any key to close the window" << std::endl;

  for (const auto &landmark : muller_landmarks) {
    cv::circle(image,
               cv::Point(static_cast<int>(landmark.first * image.cols),
                         static_cast<int>(landmark.second * image.rows)),
               2, cv::Scalar(0, 0, 255), -1);
  }

  // 이미지 표시
  cv::imshow("Muller", image);

  // 키 입력 대기
  cv::waitKey(0);

  // 윈도우 닫기
  cv::destroyAllWindows();

  return 0;
}
