
#include <vector>
//#include <iostream>
#include <functional>

#include "boost_math/boost/math/tools/numerical_differentiation.hpp"

#include "eigen-eigen-323c052e1731/Eigen/Dense"
#include "eigen-eigen-323c052e1731/Eigen/Eigen"
#include "eigen-eigen-323c052e1731/Eigen/Sparse"
#include "eigen-eigen-323c052e1731/Eigen/SparseLU"

typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SparseBlock;
typedef std::vector<int> Indicies;
typedef std::vector<Eigen::MatrixXd> DataBlocks;

double derivative(std::function<double(double)> func, double x, int order);
void SparseAssembler(SparseBlock& mat, Indicies& rows, Indicies& cols, DataBlocks& data);


