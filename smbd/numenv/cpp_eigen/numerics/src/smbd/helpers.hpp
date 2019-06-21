#pragma once

#include <vector>
//#include <iostream>
#include <functional>

#include "boost/math/tools/numerical_differentiation.hpp"

#include "eigen/Eigen/Dense"
#include "eigen/Eigen/Eigen"
#include "eigen/Eigen/Sparse"
#include "eigen/Eigen/SparseLU"

typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SparseBlock;
typedef std::vector<int> Indicies;
typedef std::vector<Eigen::MatrixXd> DataBlocks;

double derivative(std::function<double(double)> func, double x, int order);
void SparseAssembler(SparseBlock& mat, Indicies& rows, Indicies& cols, DataBlocks& data);


