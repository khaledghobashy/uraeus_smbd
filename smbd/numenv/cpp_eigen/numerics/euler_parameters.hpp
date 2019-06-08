#pragma once

#include "eigen-eigen-323c052e1731/Eigen/Dense"
#include "eigen-eigen-323c052e1731/Eigen/Eigen"

using namespace Eigen;

Matrix3d A(const Vector4d &P);
MatrixXd B(const Vector4d &P, const Vector3d &u);
