#pragma once

#include <cmath>

struct Mat4 {
    float values[16]{};

    float& at(int row, int column) { return values[column * 4 + row]; }
    float at(int row, int column) const { return values[column * 4 + row]; }
};

inline Mat4 identity() {
    Mat4 m;
    for (int i = 0; i < 4; ++i) m.at(i, i) = 1.0f;
    return m;
}

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 result;
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            for (int k = 0; k < 4; ++k)
                result.at(row, column) += a.at(row, k) * b.at(k, column);
    return result;
}

inline Mat4 translation(float x, float y, float z) {
    Mat4 m = identity();
    m.at(0, 3) = x;
    m.at(1, 3) = y;
    m.at(2, 3) = z;
    return m;
}

inline Mat4 scaling(float scale) {
    Mat4 m = identity();
    m.at(0, 0) = m.at(1, 1) = m.at(2, 2) = scale;
    return m;
}

inline Mat4 rotationX(float angle) {
    Mat4 m = identity();
    m.at(1, 1) = m.at(2, 2) = std::cos(angle);
    m.at(2, 1) = std::sin(angle);
    m.at(1, 2) = -std::sin(angle);
    return m;
}

inline Mat4 rotationY(float angle) {
    Mat4 m = identity();
    m.at(0, 0) = m.at(2, 2) = std::cos(angle);
    m.at(0, 2) = std::sin(angle);
    m.at(2, 0) = -std::sin(angle);
    return m;
}

inline Mat4 rotationZ(float angle) {
    Mat4 m = identity();
    m.at(0, 0) = m.at(1, 1) = std::cos(angle);
    m.at(1, 0) = std::sin(angle);
    m.at(0, 1) = -std::sin(angle);
    return m;
}

inline Mat4 perspective(float fov, float aspect, float near_plane, float far_plane) {
    Mat4 m;
    const float f = 1.0f / std::tan(fov / 2.0f);
    m.at(0, 0) = f / aspect;
    m.at(1, 1) = -f;
    m.at(2, 2) = far_plane / (near_plane - far_plane);
    m.at(2, 3) = far_plane * near_plane / (near_plane - far_plane);
    m.at(3, 2) = -1.0f;
    return m;
}

inline Mat4 orthographic(float half_height, float aspect, float near_plane, float far_plane) {
    Mat4 m = identity();
    m.at(0, 0) = 1.0f / (half_height * aspect);
    m.at(1, 1) = -1.0f / half_height;
    m.at(2, 2) = 1.0f / (near_plane - far_plane);
    m.at(2, 3) = near_plane / (near_plane - far_plane);
    return m;
}
