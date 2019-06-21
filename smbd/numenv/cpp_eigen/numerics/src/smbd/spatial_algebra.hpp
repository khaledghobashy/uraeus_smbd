# pragma once

#include <vector>
#include <numeric>
#include "eigen/Eigen/Dense"
#include "eigen/Eigen/Eigen"

typedef Eigen::Block<Eigen::Matrix<double, 3, 3, 0, 3, 3>, -1, -1, false> Col; 

Eigen::Matrix3d skew(Eigen::Vector3d &v);
Eigen::Vector3d orthogonal_vector(Eigen::Vector3d &v1);

template <typename T>
Eigen::Matrix3d triad(Eigen::Vector3d &v1, T const &v2);

Eigen::Matrix3d triad(Eigen::Vector3d &v1, Eigen::Vector3d &v2);
Eigen::Matrix3d triad(Eigen::Vector3d &v1, Col const &v2);
Eigen::Matrix3d triad(Eigen::Vector3d &v1);

Eigen::Vector3d mirrored(Eigen::Vector3d &v);
Eigen::Vector3d centered(std::vector<Eigen::Vector3d> const &args);
Eigen::Vector3d oriented(std::vector<Eigen::Vector3d> const &args);
