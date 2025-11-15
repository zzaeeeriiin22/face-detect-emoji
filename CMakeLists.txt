cmake_minimum_required(VERSION 3.14)
project(FaceDetectEmoji)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Homebrew 등으로 설치된 OpenCV 찾기
find_package(OpenCV REQUIRED core imgproc imgcodecs highgui)

# 실행 파일 생성
add_executable(face_detect main.cpp)

# OpenCV include 디렉토리 추가
target_include_directories(face_detect PRIVATE ${OpenCV_INCLUDE_DIRS})

# OpenCV 라이브러리 링크
target_link_libraries(face_detect PRIVATE ${OpenCV_LIBS})

# 컴파일 경고 활성화
if(MSVC)
    target_compile_options(face_detect PRIVATE /W4)
else()
    target_compile_options(face_detect PRIVATE -Wall -Wextra -Wpedantic)
endif()

# 빌드 타입이 지정되지 않은 경우 Release로 설정
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
