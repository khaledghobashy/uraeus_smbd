#include <vector>

#include "eigen-eigen-323c052e1731/Eigen/Dense"
#include "eigen-eigen-323c052e1731/Eigen/Eigen"
#include "eigen-eigen-323c052e1731/Eigen/Sparse"

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SparseBlock;
typedef std::vector<int> Indicies;
typedef std::vector<Eigen::MatrixXd> DataBlocks;

SparseBlock SparseAssembler(const int mrows, const int ncols, Indicies& rows, Indicies& cols, DataBlocks& data);