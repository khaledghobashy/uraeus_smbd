#include <fstream>
#include <iostream>
#include <vector>
#include "eigen/Eigen/Dense"

void ExportResultsCSV(const std::string& name, const std::vector<Eigen::VectorXd>& results);
