#pragma once

#include <vector>
#include <cstdint>

struct Point {
    double x;
    double y;
};

struct Image {
    int width;
    int height;
    int channels;
    std::vector<uint8_t> data;

    Image() : width(0), height(0), channels(0) {}
    Image(int w, int h, int c)
        : width(w), height(h), channels(c), data(w * h * c, 0) {}

    inline uint8_t getPixelSafe(int x, int y, int c) const {
        if (x < 0 || x >= width || y < 0 || y >= height || c < 0 || c >= channels) {
            return 0;
        }
        return data[(y * width + x) * channels + c];
    }
};
