#pragma once

#include "eigen-eigen-323c052e1731/Eigen/Dense"
#include "eigen-eigen-323c052e1731/Eigen/Eigen"

Eigen::Matrix3d A(const Eigen::Vector4d& P);
Eigen::Matrix<double, 3, 4> B(const Eigen::Vector4d& P, const Eigen::Vector3d& u);
