#pragma once

#include "eigen/Eigen/Dense"
#include "eigen/Eigen/Eigen"

Eigen::Matrix3d A(const Eigen::Vector4d& P);
Eigen::Matrix<double, 3, 4> B(const Eigen::Vector4d& P, const Eigen::Vector3d& u);
Eigen::Vector4d A2P(Eigen::Matrix3d& dcm);
